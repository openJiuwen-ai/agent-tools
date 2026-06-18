# 工作器发现配置文档

## 概述

Jiuwen Agent Router 支持多种工作器发现机制，包括：

1. **配置文件发现**：从JSON或YAML配置文件读取工作器信息
2. **etcd发现**：从etcd服务注册中心动态发现工作器

本文档主要介绍配置文件发现方式的使用方法。

## 配置文件发现

### 支持的文件格式

配置文件发现支持两种格式：

- **JSON格式**：文件扩展名为 `.json`
- **YAML格式**：文件扩展名为 `.yaml` 或 `.yml`

### 配置文件结构

配置文件需要包含一个 `workers` 数组，每个工作器包含以下字段：

| 字段名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| `worker_id` | string | 是 | 工作器唯一标识符 |
| `model` | string | 是 | 模型名称 |
| `url` | string | 是 | 工作器API地址 |
| `available_memory` | integer | 否 | 可用内存（字节），默认为0 |
| `current_load` | integer | 否 | 当前负载，默认为0 |
| `cached_prefixes` | array | 否 | 已缓存的KV缓存前缀列表，默认为空数组 |

### JSON格式示例

创建 `workers.json` 文件：

```json
{
  "workers": [
    {
      "worker_id": "worker-1",
      "model": "llama-3-70b",
      "url": "http://localhost:8001/v1",
      "available_memory": 1000000,
      "current_load": 10,
      "cached_prefixes": ["prefix-1", "prefix-2"]
    },
    {
      "worker_id": "worker-2",
      "model": "llama-3-70b",
      "url": "http://localhost:8002/v1",
      "available_memory": 800000,
      "current_load": 5,
      "cached_prefixes": ["prefix-2", "prefix-3"]
    },
    {
      "worker_id": "worker-3",
      "model": "llama-3-70b",
      "url": "http://localhost:8003/v1",
      "available_memory": 1200000,
      "current_load": 15,
      "cached_prefixes": ["prefix-1", "prefix-3"]
    }
  ]
}
```text

### YAML格式示例

创建 `workers.yaml` 文件：

```yaml
workers:
  - worker_id: worker-1
    model: llama-3-70b
    url: http://localhost:8001/v1
    available_memory: 1000000
    current_load: 10
    cached_prefixes:
      - prefix-1
      - prefix-2

  - worker_id: worker-2
    model: llama-3-70b
    url: http://localhost:8002/v1
    available_memory: 800000
    current_load: 5
    cached_prefixes:
      - prefix-2
      - prefix-3

  - worker_id: worker-3
    model: llama-3-70b
    url: http://localhost:8003/v1
    available_memory: 1200000
    current_load: 15
    cached_prefixes:
      - prefix-1
      - prefix-3
```text

## 配置方式

### 环境变量配置

通过环境变量配置工作器发现：

```bash
# 设置发现类型为配置文件方式
export WORKER_DISCOVERY_TYPE=config

# 设置配置文件路径（支持JSON或YAML）
export WORKER_CONFIG_PATH=/path/to/workers.yaml
```text

### 配置文件配置

在 `.env` 文件中配置：

```env
WORKER_DISCOVERY_TYPE=config
WORKER_CONFIG_PATH=workers.yaml
```text

## 依赖要求

### JSON格式

JSON格式是Python标准库的一部分，无需额外安装依赖。

### YAML格式

YAML格式需要安装 PyYAML 包：

```bash
pip install pyyaml
```text

## 动态更新

配置文件发现支持动态更新：

- Router 会定期（默认30秒）重新读取配置文件
- 如果配置文件发生变化，Router 会自动更新工作器列表
- 新增的工作器会被添加到可用列表
- 移除的工作器会从列表中删除

## 错误处理

配置文件发现机制包含完善的错误处理：

1. **文件不存在**：记录警告日志，返回空列表
2. **格式错误**：记录错误日志，返回空列表
3. **字段缺失**：记录错误日志，跳过该工作器
4. **依赖缺失**：YAML格式需要安装 PyYAML，否则会抛出异常

## 最佳实践

1. **使用YAML格式**：YAML格式更易读，更适合人工编辑
2. **版本控制**：将配置文件纳入版本控制系统
3. **配置验证**：在部署前验证配置文件的正确性
4. **监控更新**：监控配置文件的变化，确保及时发现配置错误

## 示例项目

项目提供了两个示例配置文件：

- `workers.json.example`：JSON格式示例
- `workers.yaml.example`：YAML格式示例

可以复制这些文件作为配置模板：

```bash
# 使用JSON格式
cp workers.json.example workers.json

# 或使用YAML格式
cp workers.yaml.example workers.yaml
```text

## 与etcd发现对比

| 特性 | 配置文件发现 | etcd发现 |
|------|-------------|---------|
| 部署复杂度 | 低 | 高 |
| 动态更新 | 定期轮询 | 实时监听 |
| 高可用性 | 单点 | 集群 |
| 适用场景 | 小型部署、测试环境 | 大型部署、生产环境 |
| 依赖 | 无（JSON）/ PyYAML（YAML） | etcd3 |
