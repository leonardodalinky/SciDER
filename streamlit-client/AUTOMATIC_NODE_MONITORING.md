# 自动 Node 中间输出监控

## 概述

现在所有 workflow 中的 node 都会自动记录中间输出，并在 Streamlit 界面中显示。无需手动修改每个 node 函数。

## 实现方式

### 1. Node Monitor Wrapper (`node_monitor_wrapper.py`)

创建了一个通用的 node 包装器，可以：
- 自动捕获 node 执行前后的状态快照
- 记录新增的消息
- 记录 node 历史
- 捕获错误信息

### 2. 修改 Build 函数

修改了以下 agent 的 `build()` 函数，自动包装所有 node：
- ✅ `scievo/agents/data_agent/build.py` - Data Agent
- ✅ `scievo/agents/experiment_agent/build.py` - Experiment Agent
- ✅ `scievo/agents/experiment_agent/exec_subagent/build.py` - Execution Subagent

### 3. 工作原理

在 `build()` 函数中，我们使用 `add_monitored_node()` 辅助函数来添加 node：

```python
def add_monitored_node(name: str, node_func, agent_name: str = "Data Agent", phase=None):
    if _MONITORING_ENABLED and wrap_node_for_monitoring:
        wrapped_func = wrap_node_for_monitoring(
            node_func,
            node_name=name,
            agent_name=agent_name,
            phase=phase or PhaseType.DATA_EXECUTION
        )
        g.add_node(name, wrapped_func)
    else:
        g.add_node(name, node_func)
```

这样，如果 `streamlit-client` 目录存在且可以导入监控模块，所有 node 都会被自动包装。如果不存在，代码会正常执行，只是不会记录监控信息。

## 记录的中间输出

每个 node 的中间输出包含：

1. **State Before**: 执行前的状态快照
   - 消息数量
   - Node 历史
   - 工作空间路径
   - 剩余计划数
   - 其他状态字段

2. **State After**: 执行后的状态快照
   - 同样的字段，但反映执行后的状态

3. **Messages Added**: 新增的消息列表
   - 消息索引
   - 消息预览

4. **Node History**: Node 执行历史
   - 显示 node 的执行路径

5. **其他指标**:
   - 消息总数
   - 剩余计划数
   - 工作空间路径

## 在 Streamlit 中查看

1. **启动工作流**后，点击 **"💬 View Conversations"** 按钮
2. 在 dialog 中，你会看到：
   - **Agent 层级**：每个 agent 的对话记录
   - **Node 层级**：每个 agent 下的各个 node
   - **Intermediate Output**：点击消息下方的 "📋 Intermediate Output" 展开查看详细信息

## 显示层级结构

```
🤖 Data Agent (10 nodes, 45 messages)
  ├─ ⚙️ planner (2 messages)
  │   ├─ Message 1
  │   │   └─ 📋 Intermediate Output
  │   │       ├─ 📥 State Before Execution
  │   │       ├─ 📤 State After Execution
  │   │       ├─ 💬 Messages Added
  │   │       └─ 🔄 Node History
  │   └─ Message 2
  ├─ ⚙️ llm_chat (15 messages)
  │   └─ ...
  └─ ⚙️ tool_calling (8 messages)
      └─ ...
```

## 添加更多 Agent 的监控

如果要为其他 agent 添加自动监控，只需修改对应的 `build()` 函数：

1. 在文件顶部添加导入代码：
```python
try:
    import sys
    from pathlib import Path
    streamlit_client_path = Path(__file__).parent.parent.parent.parent / "streamlit-client"
    if streamlit_client_path.exists():
        sys.path.insert(0, str(streamlit_client_path))
        from node_monitor_wrapper import wrap_node_for_monitoring
        from workflow_monitor import PhaseType
        _MONITORING_ENABLED = True
    else:
        _MONITORING_ENABLED = False
        wrap_node_for_monitoring = None
except ImportError:
    _MONITORING_ENABLED = False
    wrap_node_for_monitoring = None
```

2. 添加辅助函数：
```python
def add_monitored_node(name: str, node_func, agent_name: str = "Agent Name", phase=None):
    if _MONITORING_ENABLED and wrap_node_for_monitoring:
        wrapped_func = wrap_node_for_monitoring(
            node_func,
            node_name=name,
            agent_name=agent_name,
            phase=phase or PhaseType.DATA_EXECUTION
        )
        g.add_node(name, wrapped_func)
    else:
        g.add_node(name, node_func)
```

3. 替换所有 `g.add_node()` 调用为 `add_monitored_node()`

## 注意事项

1. **可选功能**：监控功能是可选的，如果 `streamlit-client` 目录不存在或无法导入，代码会正常执行，只是不会记录监控信息
2. **性能影响**：监控会捕获状态快照，可能会有轻微的性能影响，但通常可以忽略
3. **错误处理**：如果状态捕获失败，会记录错误信息，但不会影响 node 的正常执行

## 未来扩展

可以进一步扩展的功能：
- 添加更多状态字段的捕获
- 支持自定义状态快照函数
- 添加性能指标（执行时间等）
- 支持过滤敏感信息
