# 快速开始指南

欢迎开始学习 SGLang 源码！这个指南将帮助你快速上手。

## 学习系统说明

本项目提供了一套结构化的学习笔记系统：

```
sglang-learning/
├── README.md              # 项目总览和学习路线
├── QUICK_START.md         # 本文件 - 快速开始
├── LEARNING_LOG.md        # 每日学习日志
├── QUESTIONS.md           # 问题追踪
├── notes/                 # 按模块组织的学习笔记
│   ├── 00-项目概述.md
│   ├── 01-架构设计.md
│   ├── 03-managers.md
│   ├── 05-mem-cache.md
│   ├── 07-constrained.md
│   └── 10-lang-api.md
└── sglang-0.5.7/         # 源码
```

## 推荐学习路径

### 第一周：基础理解

**目标**: 了解项目整体架构和核心概念

1. **阅读官方文档** (1-2天)
   - 浏览 `sglang-0.5.7/README.md`
   - 阅读 `sglang-0.5.7/docs/` 目录下的文档
   - 运行几个 `examples/` 中的示例

2. **理解项目结构** (1天)
   - 填充 `notes/00-项目概述.md`
   - 了解各个目录的作用

3. **追踪简单请求** (2-3天)
   - 从 `launch_server.py` 开始
   - 追踪一个简单的 HTTP 请求流程
   - 记录在 `notes/02-entrypoints.md`

### 第二周：核心机制

**目标**: 深入理解调度和执行机制

1. **Scheduler 系统** (2-3天)
   - 阅读 `srt/managers/` 代码
   - 理解批处理和调度算法
   - 完善 `notes/03-managers.md`

2. **Model Executor** (2-3天)
   - 阅读 `srt/model_executor/` 代码
   - 理解模型推理流程
   - 完善 `notes/04-model-executor.md`

### 第三周：特色功能

**目标**: 理解 SGLang 的创新点

1. **RadixAttention** (3-4天)
   - 重点！这是 SGLang 的核心创新
   - 阅读 `srt/mem_cache/` 代码
   - 画出数据结构图
   - 完善 `notes/05-mem-cache.md`

2. **约束生成** (2-3天)
   - 阅读 `srt/constrained/` 代码
   - 理解 FSM 实现
   - 完善 `notes/07-constrained.md`

### 第四周及以后：高级特性

根据兴趣选择：
- 分布式系统
- 多模态支持
- JIT 内核
- Language API

## 学习方法建议

### 1. 主动追踪代码

不要只是阅读代码，要实际运行和调试：

```bash
# 进入源码目录
cd sglang-0.5.7

# 设置开发环境
pip install -e ".[dev]"

# 运行示例
python examples/quick_start.py

# 使用调试器
python -m pdb python/sglang/launch_server.py --model-path <model>
```

### 2. 记录和总结

每天学习后：
1. 在 `LEARNING_LOG.md` 中记录进度
2. 在对应的笔记文件中补充细节
3. 在 `QUESTIONS.md` 中记录疑问

### 3. 画图理解

对于复杂的流程和数据结构，建议画图：
- 流程图
- 类图
- 序列图
- 数据结构图

可以使用：
- 纸笔
- Draw.io
- Mermaid (可以直接在 Markdown 中使用)

### 4. 对比学习

对比 SGLang 与其他框架（vLLM, TGI）：
- 相同的问题，不同的解决方案
- 设计权衡
- 性能差异

### 5. 动手实验

尝试：
- 修改代码，观察效果
- 添加日志，追踪执行
- 写单元测试，验证理解
- 实现小功能

## 关键文件快速索引

### 入口点
- `python/sglang/launch_server.py` - 服务启动
- `python/sglang/srt/entrypoints/openai_api.py` - OpenAI API

### 核心调度
- `python/sglang/srt/managers/scheduler.py` - 调度器
- `python/sglang/srt/managers/tp_worker.py` - 工作进程

### 模型执行
- `python/sglang/srt/model_executor/model_runner.py` - 模型运行器
- `python/sglang/srt/model_executor/forward_batch_info.py` - 批处理信息

### RadixAttention
- `python/sglang/srt/mem_cache/radix_cache.py` - Radix缓存
- `python/sglang/srt/mem_cache/memory_pool.py` - 内存池

### 约束生成
- `python/sglang/srt/constrained/fsm_cache.py` - FSM缓存
- `python/sglang/srt/constrained/jump_forward.py` - 跳跃前向

## 学习资源

### 官方资源
- GitHub: https://github.com/sgl-project/sglang
- 文档: https://sgl-project.github.io/
- 论文: 查看项目 README 中的引用

### 相关技术
- vLLM: 了解 PagedAttention
- FlashAttention: 注意力机制优化
- Triton: GPU 内核编程

### 社区
- GitHub Issues 和 Discussions
- Discord/Slack 社区（如果有）

## 常见问题

### Q: 代码量很大，从哪里开始？
A: 从入口点开始，追踪一个简单的请求流程。不要试图一次理解所有代码。

### Q: 遇到不懂的概念怎么办？
A: 记录在 `QUESTIONS.md` 中，继续往下学。很多概念在后续学习中会逐渐清晰。

### Q: 如何验证自己的理解？
A: 尝试用自己的话解释给别人听（或写在笔记里），或者修改代码看是否符合预期。

### Q: 学习进度慢怎么办？
A: 源码学习本来就需要时间。重要的是持续学习，每天进步一点点。

## 开始学习

现在，你可以：

1. ✅ 阅读 `README.md` 了解整体学习路线
2. ✅ 选择一个起点（推荐从 `notes/00-项目概述.md` 开始）
3. ✅ 在 `LEARNING_LOG.md` 中记录你的第一天

祝学习愉快！🚀
