import json
import uuid
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union, get_args, get_origin

# import inquirer
# from prompt_toolkit import PromptSession
# from prompt_toolkit.completion import Completer, Completion
# from prompt_toolkit.formatted_text import HTML
# from prompt_toolkit.styles import Style
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .registry import registry


def debug_print_swarm(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def print_in_box(
    text: str, console: Optional[Console] = None, title: str = "", color: str = "white"
) -> None:
    """
    Print the text in a box.
    :param text: the text to print.
    :param console: the console to print the text.
    :param title: the title of the box.
    :param color: the border color.
    :return:
    """
    console = console or Console()

    # panel = Panel(text, title=title, border_style=color, expand=True, highlight=True)
    # console.print(panel)
    console.print("_" * 20 + title + "_" * 20, style=f"bold {color}")
    console.print(text, highlight=True, emoji=True)


def debug_print(debug: bool, *args: str, **kwargs: dict) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = "\n".join(map(str, args))
    color = kwargs.get("color", "white")
    title = kwargs.get("title", "")
    log_str = f"[{timestamp}]\n{message}"
    print_in_box(log_str, color=color, title=title)  # type: ignore
    log_path = kwargs.get("log_path", None)
    if log_path:
        with open(log_path, "a") as f:  # type: ignore
            f.write(log_str + "\n")


def ask_text(
    question: str, title: str = "User", console: Optional[Console] = None, default_answer: str = ""
) -> str:
    """
    Display a question in a panel and prompt the user for an answer.
    :param question: the question to display.
    :param title: the title of the panel.
    :param console: the console to use.
    :return: the user's answer.
    """
    console = console or Console()

    console.print(Panel(question, title=title, border_style="green"))
    answer = Prompt.ask(
        f"Type your answer here, press Enter to use default answer", default=default_answer
    )
    console.print(Panel(answer, title=title))
    return answer


def print_markdown(md_path: str, console: Optional[Console] = None):
    console = console or Console()
    with open(md_path, "r") as f:
        md_content = f.read()
    console.print(Markdown(md_content))


# def single_select_menu(options, message: str = ""):
#     questions = [
#         inquirer.List(
#             "choice",
#             message=message,
#             choices=options,
#         ),
#     ]
#     answers = inquirer.prompt(questions)
#     return answers["choice"]


def get_user_confirmation(prompt: str) -> bool | None:
    user_input = prompt.strip().lower()
    if user_input in ["y", "yes", "true", "t"]:
        return True
    elif user_input in ["n", "no", "false", "f"] or user_input == "":
        return False
    else:
        print("Invalid input. Please enter 'y' for yes or 'n' for no.")


def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def get_type_info(annotation, base_type_map):
    # 处理基本类型
    if annotation in base_type_map:
        return {"type": base_type_map[annotation]}

    # 处理typing类型
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)

        # 处理List类型
        if origin is list or origin is List:
            item_type = args[0]
            return {"type": "array", "items": get_type_info(item_type, base_type_map)}

        # 处理Dict类型
        elif origin is dict or origin is Dict:
            key_type, value_type = args
            if key_type != str:
                raise ValueError("Dictionary keys must be strings")

            # 如果value_type是TypedDict或Pydantic模型
            if hasattr(value_type, "__annotations__") or (
                isinstance(value_type, type) and issubclass(value_type, BaseModel)
            ):
                return get_type_info(value_type, base_type_map)

            # 普通Dict类型
            return {
                "type": "object",
                "additionalProperties": get_type_info(value_type, base_type_map),
            }

        # 处理Union类型
        elif origin is Union:
            types = [get_type_info(arg, base_type_map) for arg in args if arg != type(None)]
            if len(types) == 1:
                return types[0]
            return {"oneOf": types}

    # 处理Pydantic模型
    if isinstance(annotation, type):  # 先检查是否是类型
        try:
            if issubclass(annotation, BaseModel):
                schema = annotation.model_json_schema()
                # 提取主要schema部分
                properties = schema.get("properties", {})
                required = schema.get("required", [])

                return {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }
        except TypeError:
            pass

    # 处理dataclass
    if is_dataclass(annotation):
        properties = {}
        required = []
        for field in fields(annotation):
            properties[field.name] = get_type_info(field.type, base_type_map)
            if field.default == field.default_factory == MISSING:
                required.append(field.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    # 处理TypedDict
    if hasattr(annotation, "__annotations__"):
        properties = {}
        required = getattr(annotation, "__required_keys__", annotation.__annotations__.keys())

        for key, field_type in annotation.__annotations__.items():
            properties[key] = get_type_info(field_type, base_type_map)

        return {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        }

    # 默认返回string类型
    return {"type": "string"}


def function_to_json(func: Callable) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    ret = registry.tools_info[func.__name__].json_schema
    if ret is None:
        raise ValueError(f"Function {func.__name__} has no json schema")
    return ret


def make_tool_message(tools: Callable, args: dict, tool_content: str) -> List[Dict]:
    tool_calls = [
        {
            "type": "function",
            "function": {"name": tools.__name__, "arguments": json.dumps(args)},
            "id": str(uuid.uuid4()).replace("-", "")[:9],
        }
    ]
    return [
        {"role": "assistant", "tool_calls": tool_calls},
        {
            "role": "tool",
            "content": tool_content,
            "name": tools.__name__,
            "tool_call_id": tool_calls[0]["id"],
        },
    ]


def make_message(role: str, content: str):
    return [{"role": role, "content": content}]


# class UserCompleter(Completer):

#     def __init__(self, users: List[str]):
#         super().__init__()
#         self.users = users

#     def get_completions(self, document, complete_event):
#         word = document.get_word_before_cursor()

#         if word.startswith("@"):
#             prefix = word[1:]  # 去掉@
#             for user in self.users:
#                 if user.startswith(prefix):
#                     yield Completion(
#                         user, start_position=-len(prefix), style="fg:blue bold"  # 蓝色加粗
#                     )


def pretty_print_messages(message, **kwargs) -> None:
    # for message in messages:
    if message["role"] != "assistant" and message["role"] != "tool":
        return
    console = Console()
    if message["role"] == "tool":
        console.print("[bold blue]tool execution:[/bold blue]", end=" ")
        console.print(f"[bold purple]{message['name']}[/bold purple], result: {message['content']}")
        log_path = kwargs.get("log_path", None)
        if log_path:
            with open(log_path, "a") as file:
                file.write(f"tool execution: {message['name']}, result: {message['content']}\n")
        return

    # print agent name in blue
    console.print(f"[bold blue]{message['sender']}[/bold blue]:", end=" ")

    # print response, if any
    if message["content"]:
        console.print(message["content"], highlight=True, emoji=True)

    # print tool calls in purple, if any
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) > 1:
        console.print()
    for tool_call in tool_calls:
        f = tool_call["function"]
        name, args = f["name"], f["arguments"]
        arg_str = json.dumps(json.loads(args)).replace(":", "=")
        console.print(f"[bold purple]{name}[/bold purple]({arg_str[1:-1]})")
    log_path = kwargs.get("log_path", None)
    if log_path:
        with open(log_path, "a") as file:
            file.write(f"{message['sender']}: {message['content']}\n")
            for tool_call in tool_calls:
                f = tool_call["function"]
                name, args = f["name"], f["arguments"]
                arg_str = json.dumps(json.loads(args)).replace(":", "=")
                file.write(f"{name}({arg_str[1:-1]})\n")
