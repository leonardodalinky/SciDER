# Standard library imports
import contextlib
import copy
import json
import sys
from collections import defaultdict
from typing import Any, Callable

# Local imports
import litellm.types.utils
from httpx import ConnectError, RemoteProtocolError
from litellm import completion
from litellm.exceptions import APIError, BadRequestError, ContextWindowExceededError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scievo.tools.plan_tool import create_plans, pop_agent, set_plan_answer_and_next_step

from ..memory.utils import decode_tokens_by_tiktoken, encode_string_by_tiktoken
from .constant import API_BASE_URL, API_KEY
from .logger import LoggerManager, MetaChainLogger
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessageToolCall,
    Function,
    Message,
    Response,
    Result,
)
from .util import function_to_json, pretty_print_messages

# litellm.set_verbose=True
# litellm.num_retries = 3


def should_retry_error(retry_state: RetryCallState):
    """检查是否应该重试错误

    Args:
        retry_state: RetryCallState对象, 包含重试状态信息

    Returns:
        bool: 是否应该重试
    """
    if retry_state.outcome is None:
        return False

    exception = retry_state.outcome.exception()
    if exception is None:
        return False

    print(f"Caught exception: {type(exception).__name__} - {str(exception)}")

    # 匹配更多错误类型
    if isinstance(exception, (APIError, RemoteProtocolError, ConnectError)):
        return True

    # 通过错误消息匹配
    error_msg = str(exception).lower()
    return any(
        [
            "connection error" in error_msg,
            "server disconnected" in error_msg,
            "eof occurred" in error_msg,
            "timeout" in error_msg,
            "rate limit" in error_msg,  # 添加 rate limit 错误检查
            "rate_limit_error" in error_msg,  # Anthropic 的错误类型
            "too many requests" in error_msg,  # HTTP 429 错误
            "overloaded" in error_msg,  # 添加 Anthropic overloaded 错误
            "overloaded_error" in error_msg,  # 添加 Anthropic overloaded 错误的另一种形式
            "负载已饱和" in error_msg,  # 添加中文错误消息匹配
            "error code: 429" in error_msg,  # 添加 HTTP 429 状态码匹配
            "context_length_exceeded" in error_msg,  # 添加上下文长度超限错误匹配
        ]
    )


__CTX_VARS_NAME__ = "ctx_vars"
__HISTORY_NAME__ = "history"
logger = LoggerManager.get_logger()


def truncate_message(message: str) -> str:
    """按比例截断消息"""
    if not message:
        return message
    tokens = encode_string_by_tiktoken(message)
    # 假设每个字符平均对应1个token（这是个粗略估计）
    current_length = len(tokens)
    # 多截断一些以确保在token限制内
    max_length = 10000
    if current_length > max_length:
        return decode_tokens_by_tiktoken(tokens[:max_length])
    else:
        return message


class MetaChain:
    def __init__(self, log_path: str | MetaChainLogger | None = None):
        """
        log_path: path of log file, None
        """
        if isinstance(log_path, MetaChainLogger):
            self.logger = log_path
        elif isinstance(log_path, str):
            self.logger = MetaChainLogger(log_path=log_path)
        else:
            self.logger = MetaChainLogger(log_path=None)

        if self.logger.log_path is None:
            self.logger.info(
                "[Warning] Not specific log path, so log will not be saved",
                "...",
                title="Log Path",
                color="light_cyan3",
            )
        else:
            self.logger.info(
                "Log file is saved to",
                self.logger.log_path,
                "...",
                title="Log Path",
                color="light_cyan3",
            )

    def get_chat_completion(
        self,
        agent: Agent,
        history: list[Message],
        ctx_vars: dict,  # type: ignore
        model_override: str | None,
        debug: bool,
    ) -> litellm.types.utils.ModelResponse:
        ctx_vars: defaultdict[str, Any] = defaultdict(str, ctx_vars)
        instructions = (
            agent.instructions(ctx_vars) if callable(agent.instructions) else agent.instructions
        )

        messages = [{"role": "system", "content": instructions}] + history
        # debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "base_url": API_BASE_URL,
            "api_key": API_KEY,
        }

        if create_params["model"].startswith("mistral"):
            messages = create_params["messages"]
            for message in messages:
                if "sender" in message:
                    del message["sender"]
            create_params["messages"] = messages

        if tools and create_params["model"].startswith("gpt"):
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return completion(**create_params)  # type: ignore

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    self.logger.info(
                        error_message, title="Handle Function Result Error", color="red"
                    )
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        functions: list[AgentFunction],
        history: list[Message],
        ctx_vars: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], ctx_vars={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                self.logger.info(
                    f"Tool {name} not found in function map.", title="Tool Call Error", color="red"
                )
                new_msg = Message(
                    role="tool",
                    name=name,
                    content=f"[Tool Call Error] Error: Tool {name} not found.",
                )
                new_msg["tool_call_id"] = tool_call.id
                partial_response.messages.append(new_msg)
                continue
            kwargs = json.loads(tool_call.function.arguments)

            # debug_print(
            #     debug, f"Processing tool call: {name} with arguments {args}")
            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                kwargs[__CTX_VARS_NAME__] = ctx_vars
            if __HISTORY_NAME__ in func.__code__.co_varnames:
                kwargs[__HISTORY_NAME__] = history

            try:
                raw_result = function_map[name](**kwargs)
            except Exception as e:
                # if "case_resolved" in name:
                #     raw_result = function_map[name](tool_call.function.arguments)
                # else:
                self.logger.info(
                    f"[Tool Call Error] The execution of tool {name} failed. Error: {e}",
                    title="Tool Call Error",
                    color="red",
                )
                new_msg = Message(
                    role="tool",
                    name=name,
                    content=f"[Tool Call Error] The execution of tool {name} failed. Error: {e}",
                )
                new_msg["tool_call_id"] = tool_call.id
                partial_response.messages.append(new_msg)
                continue

            result: Result = self.handle_function_result(raw_result, debug)

            new_msg = Message(
                role="tool",
                name=name,
                content=result.value,
            )
            new_msg["tool_call_id"] = tool_call.id
            partial_response.messages.append(new_msg)
            self.logger.pretty_print_messages(partial_response.messages[-1])
            # debug_print(debug, "Tool calling: ", json.dumps(partial_response.messages[-1], indent=4), log_path=log_path, title="Tool Calling", color="green")

            partial_response.ctx_vars.update(result.ctx_vars)

        return partial_response

    def run(
        self,
        agent: Agent,
        messages: list[Message],
        ctx_vars: dict = {},
        model_override: str | None = None,
        debug: bool = True,
        max_turns: int = sys.maxsize,
        execute_tools: bool = True,
    ) -> Response:
        ctx_vars = copy.deepcopy(ctx_vars)
        ctx_vars["plan_step"] = 0
        ctx_vars["plans"] = []
        ctx_vars["plans_stack"] = []
        ctx_vars["agent_stack"] = [agent]
        ctx_vars["model"] = agent.model
        ctx_vars["forced_planning"] = True

        history = copy.deepcopy(messages)
        init_len = len(messages)

        self.logger.info(
            "Receiveing the task:", history[-1]["content"], title="Receive Task", color="green"
        )

        while len(history) - init_len < max_turns and len(ctx_vars["agent_stack"]) > 0:
            current_agent: Agent = ctx_vars["agent_stack"][-1]

            if ctx_vars["plan_step"] > len(ctx_vars["plans"]):
                if len(ctx_vars["agent_stack"]) == 1:
                    # normally plan step has reached the end
                    break
                else:
                    # TODO: pop agent
                    self.handle_tool_calls(
                        [
                            ChatCompletionMessageToolCall(
                                function=Function(name="pop_agent", arguments="{}")
                            )
                        ],
                        [pop_agent],
                        history,
                        ctx_vars,
                        debug,
                    )
                    continue

            # get completion with current history, agent
            if ctx_vars["forced_planning"]:
                hook_ctx = current_agent.hook_functions([create_plans])
            elif ctx_vars["plan_step"] == len(ctx_vars["plans"]):
                hook_ctx = current_agent.hook_functions([set_plan_answer_and_next_step])
            else:
                hook_ctx = contextlib.nullcontext()
            with hook_ctx:
                completion = self.get_chat_completion(
                    agent=current_agent,
                    history=history,
                    ctx_vars=ctx_vars,
                    model_override=model_override,
                    debug=debug,
                )
            message: Message = completion.choices[0].message  # type: ignore
            # Add a new attribute "sender" to the message
            message.sender = current_agent.name  # type: ignore
            # debug_print(
            #     debug,
            #     "Received completion:",
            #     message.model_dump_json(indent=4),
            #     log_path=log_path,
            #     title="Received Completion",
            #     color="blue",
            # )
            self.logger.pretty_print_messages(message)
            history.append(message)

            if not message.tool_calls or not execute_tools:
                self.logger.info("Ending turn.", title="End Turn", color="red")
                break

            # handle function calls, updating context_variables, and switching agents
            tool_calls = []
            for tool_call in message.tool_calls:
                # truncate tool calls after push_agent
                if tool_call.function.name == "push_agent":
                    tool_calls.append(tool_call)
                    break
                else:
                    tool_calls.append(tool_call)
            if tool_calls:
                partial_response = self.handle_tool_calls(
                    tool_calls,
                    current_agent.functions,
                    history,
                    ctx_vars,
                    debug,
                )
            else:
                partial_response = Response(messages=[message])
            history.extend(partial_response.messages)
            ctx_vars.update(partial_response.ctx_vars)

        return Response(
            messages=history[init_len:],
            ctx_vars=ctx_vars,
        )

    # TODO: 处理上下文超长的问题，需要总结性记忆
    # async def try_completion_with_truncation(
    #     self, agent, history, context_variables, model_override, stream, debug
    # ):
    #     try:
    #         return await self.get_chat_completion_async(
    #             agent=agent,
    #             history=history,
    #             context_variables=context_variables,
    #             model_override=model_override,
    #             stream=stream,
    #             debug=debug,
    #         )
    #     except (ContextWindowExceededError, BadRequestError) as e:
    #         error_msg = str(e)
    #         # 检查是否是上下文长度超限错误
    #         if "context length" in error_msg.lower() or "context_length_exceeded" in error_msg:
    #             # 提取超出的token数量
    #             # match = re.search(r'resulted in (\d+) tokens.*maximum context length is (\d+)', error_msg)
    #             # if match:
    #             # current_tokens = int(match.group(1))
    #             # max_tokens = int(match.group(2))

    #             # 修改最后一条消息
    #             if history and len(history) > 0:
    #                 last_message = history[-1]
    #                 if isinstance(last_message.get("content"), str):
    #                     last_message["content"] = truncate_message(
    #                         last_message["content"],
    #                     )
    #                     self.logger.info(
    #                         f"消息已截断以适应上下文长度限制",
    #                         title="Message Truncated",
    #                         color="yellow",
    #                     )
    #                     # 重试一次
    #                     return await self.get_chat_completion_async(
    #                         agent=agent,
    #                         history=history,
    #                         context_variables=context_variables,
    #                         model_override=model_override,
    #                         stream=stream,
    #                         debug=debug,
    #                     )
    #         # 如果不是上下文长度问题或无法处理，则重新抛出异常
    #         raise e
