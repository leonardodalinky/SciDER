# Logger 显示功能指南

## 概述

Streamlit 应用现在可以捕获并显示所有 loguru logger 的输出，让你能够实时查看工作流执行过程中的详细日志信息。

## 功能特性

### 1. **自动日志捕获**
- 自动捕获所有 loguru logger 的输出
- 支持所有日志级别：TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
- 保留日志的完整信息：时间戳、级别、模块、函数、行号、文件路径

### 2. **日志显示界面**
- 按日志级别过滤显示
- 显示日志统计信息（各级别日志数量）
- 可展开查看每条日志的详细信息
- 支持清除日志

### 3. **集成到 Workflow Monitor**
- 日志自动发送到 workflow monitor
- 可以按 agent/node 分组查看
- 日志级别自动映射到消息类型

## 使用方法

### 在 Streamlit 界面中查看日志

1. **启动工作流**后，在运行过程中或完成后：
   - 点击 **"📋 View Logs"** 按钮（在运行界面或结果页面）
   - 或者在工作流运行界面点击 **"📋 View Logs"** 按钮

2. **在日志对话框中**：
   - 使用 **"Filter by Level"** 下拉菜单过滤日志级别
   - 查看日志统计信息（各级别日志数量）
   - 点击日志条目展开查看详细信息
   - 点击 **"🗑️ Clear Logs"** 清除所有日志

3. **日志信息包括**：
   - 时间戳
   - 日志级别（带颜色标识）
   - 模块名称
   - 函数名称
   - 行号
   - 文件路径
   - 日志消息内容

## 日志级别和颜色

- ❌ **ERROR/CRITICAL**: 红色背景，用于错误和严重错误
- ⚠️ **WARNING**: 橙色背景，用于警告信息
- ✅ **SUCCESS**: 绿色背景，用于成功信息
- ℹ️ **INFO**: 蓝色背景，用于一般信息
- 🔍 **DEBUG**: 紫色背景，用于调试信息
- 🔎 **TRACE**: 灰色背景，用于跟踪信息

## 日志格式

每条日志显示格式：
```
[LEVEL] HH:MM:SS - module:function:line
日志消息内容
```

详细信息包括：
- Level: 日志级别
- Time: 时间戳
- Module: 模块名称
- Function: 函数名称
- Line: 行号
- File: 文件路径（如果可用）

## 技术实现

### Logger Handler (`logger_handler.py`)

创建了一个自定义的 loguru handler：
- `StreamlitLogHandler`: 捕获日志并存储
- `setup_streamlit_logging()`: 设置日志捕获
- `get_log_handler()`: 获取全局日志处理器实例

### 自动集成

日志捕获在应用启动时自动设置：
- 在 `main()` 函数中调用 `setup_streamlit_logging()`
- 自动移除默认 handler 并添加自定义 handler
- 同时保留控制台输出（带颜色）

### 日志限制

- 默认显示最后 500 条日志（性能考虑）
- 可以清除所有日志
- 日志存储在内存中，刷新页面会重置

## 配置选项

### 设置日志级别

在 `app_enhanced.py` 中修改：

```python
setup_streamlit_logging(min_level="DEBUG")  # 可选: TRACE, DEBUG, INFO, WARNING, ERROR
```

### 调整日志显示数量

在 `display_logs_dialog()` 函数中修改：

```python
for log in reversed(logs[-500:]):  # 修改 500 为你想要的数量
```

## 示例

### 查看所有日志
1. 运行工作流
2. 点击 "📋 View Logs"
3. 查看所有日志条目

### 只查看错误日志
1. 运行工作流
2. 点击 "📋 View Logs"
3. 在 "Filter by Level" 中选择 "ERROR"
4. 只显示错误级别的日志

### 查看特定模块的日志
1. 运行工作流
2. 点击 "📋 View Logs"
3. 展开日志条目，查看模块名称
4. 使用浏览器搜索功能（Ctrl+F / Cmd+F）搜索特定模块名

## 注意事项

1. **性能**: 大量日志可能会影响性能，默认限制为最后 500 条
2. **内存**: 日志存储在内存中，长时间运行可能会占用较多内存
3. **清除**: 刷新页面或点击 "Clear Logs" 会清除所有日志
4. **控制台输出**: 日志仍然会输出到控制台，方便调试

## 未来扩展

可以进一步扩展的功能：
- 日志导出功能（保存到文件）
- 日志搜索功能
- 日志高亮显示
- 日志统计图表
- 实时日志流显示
