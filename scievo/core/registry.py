import inspect
from dataclasses import asdict, dataclass
from typing import Callable, Literal


@dataclass
class FunctionInfo:
    name: str
    func: Callable
    args: list[str]
    docstring: str | None
    body: str
    return_type: str | None
    json_schema: dict | None

    def to_dict(self) -> dict:
        # using asdict, but exclude func field because it cannot be serialized
        d = asdict(self)
        d.pop("func")  # remove func field
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FunctionInfo":
        # if you need to create an object from a dictionary
        if "func" not in data:
            data["func"] = None  # or other default value
        return cls(**data)


class Registry:
    _instance = None
    _registry: dict[str, dict[str, Callable]] = {"tools": {}, "agents": {}}
    _registry_info: dict[str, dict[str, FunctionInfo]] = {"tools": {}, "agents": {}}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(
        self,
        type: Literal["tool", "agent"],
        name: str,
        json_schema: dict | None = None,
    ):
        """
        统一的注册装饰器
        Args:
            type: 注册类型，"tool" 或 "agent"
            name: 可选的注册名称
        """

        def decorator(func: Callable):
            nonlocal name
            if name is None:
                name = func.__name__
                # if type == "agent" and name.startswith('get_'):
                #     name = name[4:]  # 对 agent 移除 'get_' 前缀

            # 获取函数信息
            signature = inspect.signature(func)
            args = list(signature.parameters.keys())
            docstring = inspect.getdoc(func)

            # 获取函数体
            source_lines = inspect.getsource(func)
            # 移除装饰器和函数定义行
            body_lines = source_lines.split("\n")[1:]  # 跳过装饰器行
            while body_lines and (body_lines[0].strip().startswith("@") or "def " in body_lines[0]):
                body_lines = body_lines[1:]
            body = "\n".join(body_lines)

            # 获取返回类型提示
            return_type = None
            if signature.return_annotation != inspect.Signature.empty:
                return_type = str(signature.return_annotation)

            # 创建函数信息对象
            func_info = FunctionInfo(
                name=name,
                func=func,
                args=args,
                docstring=docstring,
                body=body,
                return_type=return_type,
                json_schema=json_schema,
            )

            registry_type = f"{type}s"
            self._registry[registry_type][name] = func
            self._registry_info[registry_type][name] = func_info
            return func

        return decorator

    @property
    def tools(self) -> dict[str, Callable]:
        return self._registry["tools"]

    @property
    def agents(self) -> dict[str, Callable]:
        return self._registry["agents"]

    @property
    def tools_info(self) -> dict[str, FunctionInfo]:
        return self._registry_info["tools"]

    @property
    def agents_info(self) -> dict[str, FunctionInfo]:
        return self._registry_info["agents"]


# 创建全局实例
registry = Registry()


# 便捷的注册函数
def register_tool(name: str, json_schema: dict):
    return registry.register(type="tool", name=name, json_schema=json_schema)


def register_agent(name: str):
    return registry.register(type="agent", name=name)
