# Node Intermediate Output Monitoring Guide

本指南说明如何在 Streamlit 应用中查看每个 agent 和 node 的中间输出。

## 功能概述

Streamlit 应用现在支持：
1. **按 Agent 分组显示**：所有 agent 和 subagent 的对话记录
2. **按 Node 分组显示**：每个 agent 下的各个 node 的执行记录
3. **Intermediate Output 显示**：每个 node 执行前后的状态变化、新增消息等

## 在 Agent 代码中记录 Node Intermediate Output

### 方法 1: 使用 `log_node_update()` 方法

在 agent 的 node 函数中，使用 `workflow_monitor` 记录 node 的执行和 intermediate output：

```python
from workflow_monitor import get_monitor, PhaseType

def llm_chat_node(agent_state: DataAgentState) -> DataAgentState:
    monitor = get_monitor()

    # 记录 node 开始执行
    monitor.log_node_update(
        phase=PhaseType.DATA_EXECUTION,
        node_name="llm_chat",
        status="started",
        message="Starting LLM chat node",
        agent_name="Data Agent",
        message_type="status",
    )

    # ... node 执行逻辑 ...

    # 记录 node 完成，包含 intermediate output
    monitor.log_node_update(
        phase=PhaseType.DATA_EXECUTION,
        node_name="llm_chat",
        status="completed",
        message="LLM chat node completed",
        agent_name="Data Agent",
        message_type="result",
        intermediate_output={
            "message_count": len(agent_state.history),
            "node_history": agent_state.node_history.copy(),
            "last_message_preview": agent_state.history[-1].content[:200] if agent_state.history else None,
            "workspace": str(agent_state.workspace.working_dir),
        },
    )

    return agent_state
```

### 方法 2: 使用装饰器（推荐）

创建一个装饰器来自动记录 node 的执行：

```python
from functools import wraps
from workflow_monitor import get_monitor, PhaseType

def monitor_node(node_name: str, agent_name: str, phase: PhaseType):
    """装饰器：自动记录 node 执行和 intermediate output"""
    def decorator(func):
        @wraps(func)
        def wrapper(agent_state, *args, **kwargs):
            monitor = get_monitor()

            # 记录开始
            state_before = {
                "message_count": len(agent_state.history),
                "node_history": agent_state.node_history.copy() if hasattr(agent_state, "node_history") else [],
            }

            monitor.log_node_update(
                phase=phase,
                node_name=node_name,
                status="started",
                message=f"Node '{node_name}' started",
                agent_name=agent_name,
                message_type="status",
                intermediate_output={"state_before": state_before},
            )

            try:
                # 执行 node
                result = func(agent_state, *args, **kwargs)

                # 记录完成
                state_after = {
                    "message_count": len(result.history),
                    "node_history": result.node_history.copy() if hasattr(result, "node_history") else [],
                }

                intermediate_output = {
                    "state_before": state_before,
                    "state_after": state_after,
                    "messages_added": state_after["message_count"] - state_before["message_count"],
                    "node_history": state_after["node_history"],
                }

                monitor.log_node_update(
                    phase=phase,
                    node_name=node_name,
                    status="completed",
                    message=f"Node '{node_name}' completed",
                    agent_name=agent_name,
                    message_type="result",
                    intermediate_output=intermediate_output,
                )

                return result
            except Exception as e:
                monitor.log_node_update(
                    phase=PhaseType.ERROR,
                    node_name=node_name,
                    status="error",
                    message=f"Node '{node_name}' failed: {str(e)}",
                    agent_name=agent_name,
                    message_type="error",
                    intermediate_output={"error": str(e), "state_before": state_before},
                )
                raise

        return wrapper
    return decorator

# 使用示例
@monitor_node(node_name="llm_chat", agent_name="Data Agent", phase=PhaseType.DATA_EXECUTION)
def llm_chat_node(agent_state: DataAgentState) -> DataAgentState:
    # ... node 逻辑 ...
    return agent_state
```

## Intermediate Output 数据结构

`intermediate_output` 应该是一个字典，可以包含以下字段：

```python
intermediate_output = {
    # 状态快照
    "state_before": {
        "message_count": 10,
        "node_history": ["gateway", "llm_chat"],
        "workspace": "/path/to/workspace",
    },
    "state_after": {
        "message_count": 12,
        "node_history": ["gateway", "llm_chat", "tool_calling"],
        "workspace": "/path/to/workspace",
    },

    # 变化信息
    "messages_added": [
        {"index": 11, "preview": "Message content preview..."},
        {"index": 12, "preview": "Another message..."},
    ],

    # Node 历史
    "node_history": ["gateway", "llm_chat", "tool_calling"],

    # 其他状态信息
    "message_count": 12,
    "remaining_plans_count": 3,
    "workspace": "/path/to/workspace",

    # 错误信息（如果有）
    "error": "Error message if node failed",
}
```

## 在 Streamlit 界面中查看

1. **运行工作流**后，点击 **"💬 View Conversations"** 按钮
2. 在 dialog 中，你会看到：
   - **Agent 层级**：每个 agent 的对话记录
   - **Node 层级**：每个 agent 下的各个 node
   - **Intermediate Output**：点击消息下方的 "📋 Intermediate Output" 展开查看详细信息

## 显示内容

每个 node 的 intermediate output 会显示：
- ✅ **State Before/After**：执行前后的状态快照
- 💬 **Messages Added**：新增的消息列表
- 🔄 **Node History**：node 执行历史
- 📊 **Metrics**：消息数量、剩余计划数等指标
- 📁 **Workspace**：工作空间路径
- ❌ **Errors**：错误信息（如果有）

## 注意事项

1. 如果不使用 `log_node_update()`，系统会尝试从消息和 phase 中推断 node 名称
2. Intermediate output 是可选的，如果没有提供，只会显示消息内容
3. 建议在每个 node 的开始和结束时都记录，以便更好地追踪执行流程
