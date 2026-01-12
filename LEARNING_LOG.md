# 学习日志

记录每天的学习内容和进度。

---

## 2026-01-12

### 学习内容
- ✅ 初始化学习项目
- ✅ 浏览项目结构，了解 SGLang 目录组织
- ✅ 创建学习笔记框架（12个文件）
- ✅ 创建 CLAUDE.md 为 AI 助手提供上下文
- ✅ 提交项目到 GitHub (https://github.com/bravery/sglang-learning)
- ✅ **深入学习 Pipeline Parallelism 实现原理**

### 重点学习：Pipeline Parallelism (PP)
- PP 基本原理：模型层切分、micro-batch 策略
- 配置参数：`--pp-size`, `--pp-async-batch-depth`, `--chunked-prefill-size`
- 层分配算法：自动均匀分配和自定义分配
- TP+PP 组合拓扑：理解分布式组初始化
- Stage 间通信：异步发送+同步接收设计
- 调度算法：`event_loop_pp()` 的 8 个阶段
- 性能优化：通信与计算重叠、动态 chunking
- 实战案例：DeepSeek-V3 on 96 H100 (TP=8, PP=12)

### 关键发现
- 异步通信设计避免死锁并减少流水线气泡
- Dynamic Chunking 使用二次拟合预测运行时间自动调整
- TP+PP 组合中只有 TP rank 0 负责 PP 通信，接收后 all_gather

### 关键代码位置
- `srt/managers/scheduler_pp_mixin.py` - PP 调度核心
- `srt/distributed/parallel_state.py` - 分布式状态管理
- `srt/distributed/utils.py:63-99` - 层分配算法
- `srt/model_executor/forward_batch_info.py` - PPProxyTensors

### 学习成果
- 创建了详尽的学习笔记：`notes/08-pipeline-parallelism.md` (581行)
- 理解了 SGLang 如何实现高效的流水线并行
- 掌握了 PP 与 TP 的组合使用方式

### 下一步计划
- 学习 Tensor Parallelism (TP) 的实现原理
- 研究 Expert Parallelism (EP) for MoE 模型
- 探索 Prefill-Decode Disaggregation
- 对比 SGLang PP vs vLLM PagedAttention

---

## YYYY-MM-DD

### 学习内容
-

### 关键发现
-

### 遇到的问题
-

### 下一步计划
-

