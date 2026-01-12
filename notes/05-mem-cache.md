# 05 - Memory Cache 系统

## 概述

SGLang 的 RadixAttention 是其核心创新之一，通过前缀树结构管理 KV Cache，实现高效的内存共享和复用。

**位置**: `srt/mem_cache/`

## RadixAttention 原理

### 什么是 RadixAttention?

RadixAttention 使用 Radix Tree (基数树/前缀树) 来组织和管理 KV Cache：

```
示例：多个请求共享公共前缀

Request 1: "What is the capital of France?"
Request 2: "What is the capital of Germany?"
Request 3: "What is the weather today?"

Radix Tree:
        [root]
          |
        "What is the"
        /           \
   "capital of"   "weather today?"
      /      \
  "France?"  "Germany?"
```

### 优势

1. **自动前缀共享**: 相同前缀的请求共享 KV Cache
2. **减少内存**: 避免重复存储相同的 KV 状态
3. **减少计算**: 公共前缀只需计算一次
4. **LRU 淘汰**: 支持缓存淘汰和重用

## 核心组件

### 1. Radix Cache


### 2. Tree Node


### 3. Memory Pool


## KV Cache 管理

### 分配策略


### 淘汰策略


### 共享机制


## 代码分析

### 主要文件


### 关键数据结构


### 关键算法


## 性能特征

### 1. 内存效率


### 2. 命中率


### 3. 开销分析


## 实际应用场景

### 1. Chat 对话


### 2. Few-shot Learning


### 3. 长文本处理


## 学习笔记

### 日期:

#### 今日学习内容

#### 关键发现

#### 疑问

