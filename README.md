# SGLang 源码学习笔记

本项目用于跟踪学习 SGLang (Structured Generation Language) 源码的过程。

## 项目信息
- **版本**: 0.5.7
- **开始学习日期**: 2026-01-12
- **项目仓库**: https://github.com/sgl-project/sglang

## 目录结构

```
sglang-0.5.7/
├── python/sglang/          # 主要Python包
│   ├── srt/                # SGLang Runtime - 核心运行时系统
│   ├── lang/               # 语言相关API
│   ├── jit_kernel/         # JIT编译的内核
│   ├── multimodal_gen/     # 多模态生成
│   └── cli/                # 命令行接口
├── sgl-kernel/             # C++/CUDA内核实现
├── sgl-model-gateway/      # 模型网关
├── benchmark/              # 基准测试
├── examples/               # 示例代码
├── docs/                   # 文档
└── test/                   # 测试代码
```

## 学习路线

### 1. 基础架构理解
- [ ] 项目整体架构
- [ ] 核心概念和设计理念
- [ ] 入口点和启动流程

### 2. 核心模块学习（按推荐顺序）

#### 第一阶段：理解基础组件
1. **配置与入口** (`srt/server_args.py`, `srt/entrypoints/`)
2. **模型加载** (`srt/model_loader/`)
3. **分词器** (`srt/tokenizer/`)

#### 第二阶段：核心执行流程
4. **管理器系统** (`srt/managers/`) - 请求调度和生命周期管理
5. **模型执行器** (`srt/model_executor/`) - 模型推理执行
6. **采样策略** (`srt/sampling/`) - 生成采样方法

#### 第三阶段：优化与特性
7. **内存缓存** (`srt/mem_cache/`) - KV缓存管理
8. **约束生成** (`srt/constrained/`) - 结构化输出
9. **批处理优化** (`srt/batch_overlap/`, `srt/batch_invariant_ops/`)

#### 第四阶段：高级特性
10. **分布式系统** (`srt/distributed/`)
11. **硬件后端** (`srt/hardware_backend/`)
12. **JIT内核** (`jit_kernel/`)
13. **多模态** (`srt/multimodal/`, `multimodal_gen/`)

#### 第五阶段：扩展功能
14. **LoRA支持** (`srt/lora/`)
15. **推测解码** (`srt/speculative/`)
16. **函数调用** (`srt/function_call/`)
17. **语言API** (`lang/`)

## 学习笔记

详细的模块学习笔记请查看 [notes/](./notes/) 目录：

- [00-项目概述.md](./notes/00-项目概述.md) - 项目整体介绍
- [01-架构设计.md](./notes/01-架构设计.md) - 架构设计文档
- [02-entrypoints.md](./notes/02-entrypoints.md) - 入口点分析
- [03-managers.md](./notes/03-managers.md) - 管理器系统
- [04-model-executor.md](./notes/04-model-executor.md) - 模型执行器
- [05-mem-cache.md](./notes/05-mem-cache.md) - 内存缓存系统
- [06-sampling.md](./notes/06-sampling.md) - 采样策略
- [07-constrained.md](./notes/07-constrained.md) - 约束生成
- [08-distributed.md](./notes/08-distributed.md) - 分布式系统
- [09-jit-kernel.md](./notes/09-jit-kernel.md) - JIT内核
- [10-lang-api.md](./notes/10-lang-api.md) - 语言API

## 学习资源

- 官方文档: https://sgl-project.github.io/
- 论文和博客
- 社区讨论

## 学习进度

| 模块 | 状态 | 开始日期 | 完成日期 | 笔记链接 |
|------|------|----------|----------|----------|
| 项目概述 | 未开始 | - | - | [notes/00-项目概述.md](./notes/00-项目概述.md) |
| 架构设计 | 未开始 | - | - | [notes/01-架构设计.md](./notes/01-架构设计.md) |
| 入口点 | 未开始 | - | - | [notes/02-entrypoints.md](./notes/02-entrypoints.md) |
| 管理器系统 | 未开始 | - | - | [notes/03-managers.md](./notes/03-managers.md) |
| 模型执行器 | 未开始 | - | - | [notes/04-model-executor.md](./notes/04-model-executor.md) |

## 问题和思考

记录学习过程中遇到的问题和思考，见 [QUESTIONS.md](./QUESTIONS.md)

## 学习心得

- 每个模块学习后的总结和心得
- 对比其他框架(vLLM, TGI等)的异同
- 性能优化的关键点
