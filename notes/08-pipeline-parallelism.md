# 08 - Pipeline Parallelism 流水线并行

## 概述

Pipeline Parallelism (PP) 是 SGLang 支持的重要分布式训练/推理策略，将模型的不同层分配到不同的 GPU stage 上，实现流水线式并行处理。

**位置**: `srt/distributed/`, `srt/managers/scheduler_pp_mixin.py`

## 1. 核心概念

### 什么是 Pipeline Parallelism？

PP 将模型按层切分成多个 stage，每个 stage 在不同的 GPU 上运行：

```
Stage 0 (GPU 0): Layers 0-7
Stage 1 (GPU 1): Layers 8-15
Stage 2 (GPU 2): Layers 16-23
Stage 3 (GPU 3): Layers 24-31
```

数据流：
```
Input → Stage 0 → Stage 1 → Stage 2 → Stage 3 → Output
        (hidden)  (hidden)  (hidden)  (hidden)
```

### Micro-batch 策略

为了减少流水线气泡，将一个大 batch 切分成多个 micro-batch：

```
时间轴（T0-T7）:
         T0    T1    T2    T3    T4    T5    T6    T7
Stage 0: [MB0] [MB1] [MB2] [MB3] ...
Stage 1:       [MB0] [MB1] [MB2] [MB3] ...
Stage 2:             [MB0] [MB1] [MB2] [MB3] ...
Stage 3:                   [MB0] [MB1] [MB2] [MB3] ...
```

## 2. 配置参数

### 启动参数

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 \
  --tp 8 \                          # Tensor Parallelism 度数
  --pp-size 4 \                     # Pipeline Parallelism 度数
  --pp-max-micro-batch-size 8 \    # 最大 micro-batch 大小
  --pp-async-batch-depth 2 \       # 异步 batch 深度（减少气泡）
  --chunked-prefill-size 4096 \    # 分块 prefill 大小
  --enable-dynamic-chunking         # 启用动态 chunking
```

### 核心配置（server_args.py:315-320）

```python
pp_size: int = 1                              # PP 度数
pp_max_micro_batch_size: Optional[int] = None # Micro-batch 大小
pp_async_batch_depth: int = 0                 # 异步深度
```

### 环境变量

| 变量 | 作用 | 示例 |
|------|------|------|
| `SGLANG_PP_LAYER_PARTITION` | 自定义层分配 | "10,10,12,12" for 4 stages |
| `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR` | 动态 chunk 平滑因子 | 0.75 (default) |

## 3. 层分配策略

### 自动分配算法

**文件**: `distributed/utils.py:63-99`

```python
def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int):
    """均匀分配模型层到不同 PP stage"""
    base_layers = num_hidden_layers // pp_size
    remainder = num_hidden_layers % pp_size

    # 如果层数不能整除，最后 N 个 partition 多分配一层
    if pp_rank < pp_size - remainder:
        start_layer = pp_rank * base_layers
        end_layer = start_layer + base_layers
    else:
        offset = pp_size - remainder
        start_layer = offset * base_layers + (pp_rank - offset) * (base_layers + 1)
        end_layer = start_layer + base_layers + 1

    return start_layer, end_layer
```

### 自定义分配示例

对于 32 层模型，4 个 stage：

```bash
# 均匀分配：8,8,8,8
python launch_server.py --pp-size 4

# 自定义分配：6,8,8,10
export SGLANG_PP_LAYER_PARTITION="6,8,8,10"
python launch_server.py --pp-size 4
```

## 4. 分布式组初始化

### TP + PP 组合拓扑

**文件**: `distributed/parallel_state.py:1702-1717`

对于 **8 GPU，TP=2，PP=4**：

```
物理布局:
  GPU: g0  g1  g2  g3  g4  g5  g6  g7

TP 组（张量并行，行向）:
  [g0, g1]  [g2, g3]  [g4, g5]  [g6, g7]

PP 组（流水线并行，列向）:
  [g0, g2, g4, g6]
  [g1, g3, g5, g7]
```

初始化代码：
```python
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
for i in range(num_pipeline_model_parallel_groups):
    # 创建 PP 流水线：[rank_i, rank_{i+n}, rank_{i+2n}, ...]
    ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
    group_ranks.append(ranks)
```

## 5. Stage 间通信机制

### 5.1 通信数据结构

**PPProxyTensors** (forward_batch_info.py:1131-1159):

```python
class PPProxyTensors:
    """在 PP stage 之间传递的代理张量"""
    tensors: Dict[str, torch.Tensor]

    # 典型内容：
    # {
    #   "hidden_states": Tensor([seq_len, hidden_size]),
    #   "residual": Tensor([seq_len, hidden_size]),
    # }
```

### 5.2 通信原语

**GroupCoordinator** (parallel_state.py:1183-1296):

```python
class GroupCoordinator:
    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        dst: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        async_send: bool = False,
    ) -> Optional[List[P2PWork]]:
        """异步发送张量字典到下一个 stage"""

    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """同步接收张量字典从上一个 stage"""
```

### 5.3 通信时序

**异步发送 + 同步接收**（scheduler_pp_mixin.py）:

```python
# 1. 异步发送到下一 stage
send_work = self._pp_send_dict_to_next_stage(
    tensor_dict, async_send=True
)

# 2. 继续执行其他操作（与通信重叠）
...

# 3. 稍后等待发送完成
self._pp_commit_comm_work(send_work)

# 4. 从上一 stage 同步接收（阻塞）
recv_tensors = self._pp_recv_proxy_tensors()
```

### 5.4 TP + PP 通信协调

PP 发送时只有 TP rank 0 参与：

```python
def _pp_send_pyobj_to_next_stage(self, data, async_send=False):
    if self.attn_tp_rank == 0:  # 只有 TP 第一个 rank
        dst_rank = ((self.pp_rank + 1) % self.pp_size) * self.tp_size
        p2p_work = point_to_point_pyobj(data, dst=dst_rank, async_send=async_send)
    return p2p_work
```

接收后在 TP 组内广播：
```python
tensor = self.pp_group.recv_tensor_dict(all_gather_group=self.attn_tp_group)
# all_gather_group 会自动广播给 TP 组内其他 rank
```

## 6. Micro-batch 调度算法

### 6.1 循环状态初始化

**scheduler_pp_mixin.py:506-524**:

```python
def init_pp_loop_state(self):
    # PP loop 大小 = PP 度数 + 异步深度
    self.pp_loop_size = self.pp_size + self.server_args.pp_async_batch_depth

    # 为每个 micro-batch 位置预分配资源
    self.mbs = [None] * self.pp_loop_size              # 当前批次
    self.running_mbs = [ScheduleBatch() for _ in range(self.pp_loop_size)]
    self.mb_metadata = [None] * self.pp_loop_size

    # 异步通信工作队列
    self.send_req_work = []       # 发送请求
    self.send_proxy_work = []     # 发送代理张量
    self.send_output_work = []    # 发送输出
```

### 6.2 统一调度流程

**event_loop_pp()** 主循环（scheduler_pp_mixin.py:42-140）:

```python
FOR mb_id in range(self.pp_loop_size):

    # ===== Phase 1: 接收和发送请求 =====
    recv_reqs = self.recv_requests()
    self.send_req_work = self._pp_send_pyobj_to_next_stage(
        recv_reqs, async_send=True
    )

    # ===== Phase 2: 获取批次 =====
    self.mbs[mb_id] = self.get_next_batch_to_run()

    # ===== Phase 3: 接收上一 stage 的隐藏状态 =====
    pp_proxy_tensors = self._pp_recv_proxy_tensors()  # 同步

    # ===== Phase 4: 提前处理输出（可选）=====
    if self.server_args.pp_async_batch_depth > 0:
        next_pp_outputs = self._pp_commit_send_output_work_and_preprocess()

    # ===== Phase 5: 等待之前的通信完成 =====
    self._pp_commit_comm_work(self.send_proxy_work)

    # ===== Phase 6: GPU 计算 =====
    with self.forward_stream_ctx:
        result = self.run_batch(self.mbs[mb_id], pp_proxy_tensors)
        event = torch.cuda.Event()
        event.record()

    # ===== Phase 7: 处理批次结果（CPU） =====
    next_mb_id = (mb_id + 1) % self.pp_loop_size
    self._pp_process_batch_result(self.mbs[next_mb_id], result)

    # ===== Phase 8: 异步发送隐藏状态到下一 stage =====
    self.send_proxy_work = self._pp_send_dict_to_next_stage(
        result.pp_hidden_states_proxy_tensors.tensors, async_send=True
    )
```

### 6.3 时间线图解

```
时间 →
┌─────────────────────────────────────────────────────┐
│ MB0  │ Recv Reqs │ Get Batch │ Recv Hidden │ Compute │
│      │ Send Reqs │           │             │ Send H  │
├─────────────────────────────────────────────────────┤
│ MB1  │           │ Recv Reqs │ Get Batch   │ Recv H  │
│      │           │ Send Reqs │             │ Compute │
├─────────────────────────────────────────────────────┤
│ MB2  │           │           │ Recv Reqs   │ Get B   │
│      │           │           │ Send Reqs   │ Recv H  │
└─────────────────────────────────────────────────────┘

图例：
- Recv H: 接收隐藏状态
- Send H: 发送隐藏状态
- Get B: 获取批次
- Compute: GPU 计算
```

## 7. 性能优化技术

### 7.1 通信与计算重叠

**三大重叠机制**：

1. **异步发送请求**
   ```python
   # 不等待发送完成，立即继续
   send_work = send_pyobj_to_next_stage(data, async_send=True)
   # ... 继续其他工作 ...
   commit_comm_work(send_work)  # 稍后等待
   ```

2. **GPU 计算与 CPU 处理重叠**
   ```python
   # GPU stream 中计算
   with self.forward_stream_ctx:
       result = self.run_batch(batch)

   # 同时 CPU 处理上一个批次
   self._pp_process_batch_result(last_batch, last_result)
   ```

3. **发送隐藏状态与下一批次准备重叠**
   ```python
   # 异步发送当前结果
   send_proxy_work = send_dict_to_next_stage(hidden, async_send=True)
   # 同时准备下一个批次
   next_batch = get_next_batch_to_run()
   ```

### 7.2 异步批处理深度

**pp_async_batch_depth** 参数的作用：

```
pp_async_batch_depth = 0 (默认):
  PP loop size = pp_size
  [MB0] [MB1] [MB2] [MB3]

pp_async_batch_depth = 2:
  PP loop size = pp_size + 2
  [MB0] [MB1] [MB2] [MB3] [MB4] [MB5]

效果：
- 在处理 MB4 输出时，可以提前启动 MB0
- 减少流水线气泡
- 代价：增加内存占用
```

### 7.3 动态 Chunking

**目标**：根据实际执行时间动态调整 prefill chunk 大小，使各 stage 运行时间均衡。

**预测算法**（scheduler_pp_mixin.py:525-667）:

```python
class ChunkSizePredictor:
    """使用二次函数拟合运行时间"""

    def fit(self, seq_lens: List[int], latencies: List[float]):
        # 拟合 f(l) = al² + bl + c
        self.coeff_a, self.coeff_b, self.coeff_c = np.polyfit(
            seq_lens, latencies, deg=2
        )

    def predict_next_chunk_size(
        self, history_len: int, base_chunk_size: int
    ) -> int:
        # 预测: Runtime(history_len + chunk) - Runtime(history_len)
        #     = Runtime(base_chunk_size)

        target_latency = self.predict_latency(base_chunk_size)
        current_latency = self.predict_latency(history_len)

        # 求解: f(history_len + x) - f(history_len) = target_latency
        # 即: a·x² + b·x + c·x = target_latency
        predicted_chunk = solve_quadratic(...)

        # 平滑调整
        smooth_factor = float(os.getenv("SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR", "0.75"))
        return smooth_factor * predicted_chunk + (1 - smooth_factor) * base_chunk_size
```

**使用示例**：

```bash
# 启用动态 chunking
python launch_server.py \
  --pp-size 4 \
  --chunked-prefill-size 4096 \
  --enable-dynamic-chunking

# 调整平滑因子
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.9  # 更激进
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.5  # 更保守
```

### 7.4 CUDA Graph 优化

**PP 中的 CUDA Graph 支持**（scheduler_pp_mixin.py:1035-1061）:

```python
def _pp_launch_batch(self, mb_id, pp_proxy_tensors, ...):
    with torch.profiler.record_function("run_batch"):
        with self.forward_stream_ctx:
            result = self.run_batch(batch, pp_proxy_tensors)

            # 记录是否可以运行 CUDA graph
            mb_metadata[mb_id] = PPBatchMetadata(
                can_run_cuda_graph=result.can_run_cuda_graph,
            )
```

**限制**：
- PP 中 CUDA Graph 支持有限，因为 stage 间通信会打断图
- 通常只在单个 stage 内部使用

## 8. 与其他并行策略的组合

### 8.1 TP + PP

**组合策略**（8 GPU，TP=2，PP=4）:

```
TP 组（行向）: [g0,g1] [g2,g3] [g4,g5] [g6,g7]
PP 组（列向）: [g0,g2,g4,g6] [g1,g3,g5,g7]

通信模式：
- PP 间通信：只有 TP rank 0 发送/接收
- PP 接收后 all_gather 到 TP 组内其他 rank
- TP 内通信：all_reduce, all_gather
```

### 8.2 PP + PD Disaggregation

SGLang 支持 PP 与 Prefill-Decode 分离的结合：

**三种 PP 事件循环**：

1. `event_loop_pp()` - 标准 PP
2. `event_loop_pp_disagg_prefill()` - PP + PD Prefill
3. `event_loop_pp_disagg_decode()` - PP + PD Decode

**额外通信**：
- Bootstrap requests（启动请求）
- KV transfer & consensus（KV 传输和共识）
- Release requests（释放请求）

### 8.3 PP + EP (Expert Parallelism)

**DeepSeek-V3 配置示例**：

```bash
# 96 GPU: TP=8, PP=4, EP=3
python launch_server.py \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --pp-size 4 \
  --ep-size 3
```

**通信模式**：
- PP: stage 间点对点
- TP: 组内 all_reduce
- EP: 专家路由 + 动态调度

## 9. 调试和监控

### 9.1 性能分析

```bash
# 启用 profiling
export SGLANG_ENABLE_PROFILING=1

# 查看 PP 通信时间
# 在日志中搜索：
# - "pp_send_dict"
# - "pp_recv_proxy"
# - "pp_launch_batch"
```

### 9.2 常见问题诊断

**问题 1：流水线气泡过大**
```
症状：GPU 利用率低，stage 间不均衡
解决：
- 增加 pp_async_batch_depth
- 启用 dynamic chunking
- 调整层分配（SGLANG_PP_LAYER_PARTITION）
```

**问题 2：OOM (Out of Memory)**
```
症状：CUDA out of memory
解决：
- 减小 pp_max_micro_batch_size
- 减小 pp_async_batch_depth
- 增加 pp_size（更多 stage，每个 stage 层数更少）
```

**问题 3：通信瓶颈**
```
症状：网络 I/O 饱和
解决：
- 使用更快的互连（InfiniBand, NVLink）
- 增加 TP size（减少 PP 通信量）
- 检查网络配置（NCCL_DEBUG=INFO）
```

## 10. 实战案例

### 案例 1：DeepSeek-V3 on 96 H100

**配置**：
```bash
# 节点配置
NNODES=12  # 12 个节点
GPUS_PER_NODE=8

# 并行配置
TP=8       # 每个节点内 Tensor Parallel
PP=12      # 跨节点 Pipeline Parallel

# 启动命令
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --nnodes 12 \
  --node-rank $NODE_RANK \
  --tp 8 \
  --pp-size 12 \
  --chunked-prefill-size 4096 \
  --enable-dynamic-chunking \
  --pp-async-batch-depth 2
```

**性能指标**：
- Prefill throughput: 3.8x vs 单节点
- Decode throughput: 4.8x vs 单节点
- 详见博客：https://lmsys.org/blog/2025-09-25-gb200-part-2/

### 案例 2：GB200 NVL72 Rack-scale

**拓扑**：
```
72 GPU (GB200):
- TP=8 (within NVL group)
- PP=9 (across NVL groups)
```

**特点**：
- NVLink 内部高速互连（TP）
- InfiniBand 跨组通信（PP）
- 混合 PD disaggregation

## 学习笔记

### 日期: 2026-01-12

#### 今日学习内容
- 完整理解 SGLang PP 实现原理
- 分析了层分配、通信、调度算法
- 学习了 TP+PP 组合拓扑

#### 关键发现
1. **异步发送 + 同步接收**设计巧妙，避免死锁同时减少气泡
2. **Dynamic Chunking** 使用二次拟合预测运行时间
3. **PP + TP 组合**中，TP rank 0 负责 PP 通信，然后 all_gather 给组内

#### 疑问
- [ ] PP 与 Speculative Decoding 如何结合？
- [ ] 极端不均衡模型（如 MoE）的层分配策略？
- [ ] PP 中的 KV cache 如何管理？每个 stage 都有吗？

#### 参考资料
- 源码：`srt/managers/scheduler_pp_mixin.py`
- 文档：`docs/advanced_features/pipeline_parallelism.md`
- 博客：GB200 Part 2, Large-scale EP
