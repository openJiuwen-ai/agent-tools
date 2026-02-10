# 简介
openJiuwen-vllm-affinity提供了和vllm亲和的增强能力，这些能力将通过插件的形式集成到vllm中。openJiuwen可以利用这些能力增强vllm的功能和性能。
# 特性1：Prefix双区缓存主动管理
## 背景
多轮对话总，上下文引擎会对部分历史对话进行总结压缩，或者对冗余的工具调用返回结果进行卸载，减少历史对话数据量，减少上下文的长度。
上下文的变化回导致部分kv cache失效，当前vllm使用LRU的缓存更新策略，释放的cache处于缓存的尾部，无法快速清理空间，影响整体的缓存命中率和TTFT。

## 方案
实现双区缓存和主动缓存管理。
双区缓存：缓存划分为两个区，老化区和新鲜区，老化区位于队首，新鲜区位于队尾。缓存块释放的时候，如果明确后续不会再使用该缓存块，那么把该缓存块加入老化区尾部加速老化；如果是后续可能会用到的，那么加入新鲜区的尾部，避免过早失效。

## 接口
### chat接口
chat修改如下：
增加“cache_sharing",为true时启用缓存共享，cach_salt将作为缓存的客户端标识，但是不同cache_salt的客户端将可以共享缓存块。"cache_sharing"为false时，cach_salt将避免不同的客户端共享缓存块
```RESTFul
RESTFul./v1/chat/completions
{
    "cache_salt": "1234",
    "cache_sharing": true,
    "model": "Qwen3-32B",
    "messages": [],
    ...
}
```

### 释放缓存接口
释放缓存接口如下：
```RESTFul
RESTFul./release_kv_cache
{
    "messages_released_index": 20,
    "tools_released_index": 100,
    "cache_salt": 1234,
    "cache_sharing": true,
    "model": "Qwen3-32B",
    "messages": [],
    "tools":[],
    ...
}
```
接口各字段与v1/chat/completions接口的字段基本一致，含义也一致，额外增加了messages_released_index和tools_released_index。
messages_released_index指定压缩的起始message index，也就是新的message相比旧的message从第几个message开始改变，如果没有改变，就传messages的长度。
tools_released_index指定工具变化的起始index，也就是新的tools相比旧的tools从第几个tool开始改变，如果一样则可以传tools的最大长度，或者不传该字段。

