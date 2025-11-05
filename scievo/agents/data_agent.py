"""
Agent for data understanding and processing
"""

import json

from functional import seq
from langgraph.graph import END, START, StateGraph

from scievo.core import constant
from scievo.core.llms import ModelRegistry
from scievo.core.types import GraphState, Message
from scievo.core.utils import wrap_dict_to_toon
from scievo.prompts import PROMPTS
from scievo.tools import Tool, ToolRegistry

LLM_NAME = "data"
AGENT_NAME = "data"


def gateway_node(graph_state: GraphState) -> GraphState:
    # NOTE: this node does nothing, it's just a placeholder for the conditional edges
    # Check `gateway_conditional` for the actual logic
    agent_state = graph_state.agents[AGENT_NAME]
    agent_state.round += 1
    return graph_state


def gateway_conditional(graph_state: GraphState) -> str:
    last_msg = graph_state.agents[AGENT_NAME].data_msgs[-1]
    if (tool_calls := last_msg.tool_calls) and len(tool_calls) > 0:
        return "tool_calling"

    match last_msg.role:
        case "user" | "tool":
            return "llm_chat"
        case "assistant":
            return END
        case _:
            raise ValueError(f"Unknown message role: {last_msg.role}")


def llm_chat_node(graph_state: GraphState) -> GraphState:
    agent_state = graph_state.agents[AGENT_NAME]
    selected_state = {
        "working_dir": agent_state.local_env.working_dir,
        "toolsets": agent_state.toolsets,
    }

    system_prompt = PROMPTS.data.system_prompt.format(
        state=wrap_dict_to_toon(selected_state),
        toolsets_desc=wrap_dict_to_toon(
            ToolRegistry.get_toolsets_desc(["fs"]),
        ),
    )

    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    tools.update(ToolRegistry.get_toolset("noop"))
    tools.update(ToolRegistry.get_toolset("state"))
    msg = ModelRegistry.completion(
        LLM_NAME,
        seq(agent_state.data_msgs).filter_not(lambda msg: msg.hidden).to_list(),
        system_prompt,
        agent_sender=AGENT_NAME,
        tools=[tool.name for tool in tools.values()],
    )
    agent_state.data_msgs.append(msg)
    return graph_state


def tool_calling_node(graph_state: GraphState) -> GraphState:
    """Execute tool calls from the last message and update the graph state"""
    agent_state = graph_state.agents[AGENT_NAME]
    # Get the last message which contains tool calls
    last_msg = agent_state.data_msgs[-1]

    if not last_msg.tool_calls:
        raise ValueError("No tool calls found in the last message")

    # Create a function map for available tools
    tools: dict[str, Tool] = {}
    for toolset in agent_state.toolsets:
        tools.update(ToolRegistry.get_toolset(toolset))
    tools.update(ToolRegistry.get_toolset("noop"))
    tools.update(ToolRegistry.get_toolset("state"))

    function_map = {tool.name: tool.func for tool in tools.values()}

    # Execute each tool call
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call.function.name

        # Check if tool exists in function map
        if tool_name not in function_map:
            error_msg = f"Tool {tool_name} not found"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.data_msgs.append(Message(**tool_response))
            continue

        # Parse tool arguments
        try:
            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in tool arguments: {e}"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.data_msgs.append(Message(**tool_response))
            continue
        except AssertionError as e:
            error_msg = f"Invalid tool arguments: {e}"
            tool_response = {
                "role": "tool",
                "tool_name": tool_name,
                "tool_call_id": tool_call.id,
                "content": error_msg,
            }
            agent_state.data_msgs.append(Message(**tool_response))
            continue

        # Execute the tool
        try:
            # Pass the graph state to the tool function
            func = function_map[tool_name]

            # Check if function expects graph_state parameter
            import inspect

            sig = inspect.signature(func)
            if constant.__GRAPH_STATE_NAME__ in sig.parameters:
                args.update({constant.__GRAPH_STATE_NAME__: graph_state})
            if constant.__CTX_NAME__ in sig.parameters:
                args.update({constant.__CTX_NAME__: {"current_agent": AGENT_NAME}})

            # Execute the tool in the agent's local environment
            with graph_state.agents[AGENT_NAME].local_env:
                result = func(**args)

            # Create tool response message
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": str(result),  # Ensure result is string
            }
            agent_state.data_msgs.append(Message(**tool_response))

        except Exception as e:
            error_msg = f"Tool {tool_name} execution failed: {e}"
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "content": error_msg,
            }
            agent_state.data_msgs.append(Message(**tool_response))

    return graph_state


def build():
    g = StateGraph(GraphState)

    # nodes
    g.add_node("gateway", gateway_node)
    g.add_node("llm_chat", llm_chat_node)
    g.add_node("tool_calling", tool_calling_node)

    # edges from gateway to nodes
    g.add_edge(START, "gateway")
    g.add_conditional_edges("gateway", gateway_conditional, ["llm_chat", "tool_calling", END])

    # edges from nodes to gateway
    g.add_edge("llm_chat", "gateway")
    g.add_edge("tool_calling", "gateway")

    return g
