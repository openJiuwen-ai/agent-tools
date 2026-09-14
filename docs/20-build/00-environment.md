| 字段 | 值 |
| --- | --- |
| 标题 | 环境与依赖 |
| 修改日期 | 2026-09-14 |
| 关联模块 | packages/openjiuwen_ascend_ops/setup.py, pyproject.toml |
| 关联文档 | 00-index.md, 20-build/01-build-guide.md |

# 环境与依赖

## 获取代码

catlass（CANN 开源模板库）以 git submodule 形式引入，clone 后需初始化：

```bash
git clone --recursive <本仓库地址>
# 或对已有 clone 补拉：
git submodule update --init --recursive
```

submodule 位置 `packages/openjiuwen_ascend_ops/csrc/third_party/catlass`，指向 `https://gitcode.com/cann/catlass.git` 的 `catlass-v1-stable` 分支，pin 到 commit `716fd7b`（与 vllm-ascend 一致）。构建脚本 `csrc/build.sh` 检测到缺失时也会自动执行拉取。

## 硬件要求

- **Ascend NPU**：Ascend 910B（主支持，含 910B3）/ Ascend 910_93 / Ascend 910P / Ascend 720。
- **最小显存**：每卡 32GB。

算子在 `op_host/<op>_def.cpp` 中同时注册 `ascend910_93` 与 `ascend910b` 两档 SoC。910b 用 `jitCompile.flag=static_false`（静态编译），910_93 走默认 JIT。

## 软件要求

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.11 | wheel 标签为 `cp311`；`setup.py` 分类器写 3.8/3.9/3.10 但实际只验证 3.11 |
| PyTorch | 2.9.0 | `install_requires: torch>=2.9.0` |
| torch_npu | 2.9.0 | 必须与 PyTorch ABI 对齐；adapter `#include` torch_npu 内部头，运行时 `-ltorch_npu` |
| CANN | 8.5.1 | AscendC 内核编译所需；也兼容 9.0.0 |
| setuptools | >= 61.0 | PEP 621 元数据支持 |
| wheel | 最新 | 打包 |

## 环境变量

### CANN 路径探测

`csrc/build.sh` 按以下顺序探测 CANN 安装（root/非 root 用户分桶）：

```
$ASCEND_HOME_PATH
$ASCEND_OPP_PATH 的父目录
~/Ascend/ascend-toolkit/latest（非 root）
/usr/local/Ascend/ascend-toolkit/latest（root）
```

找到后 source 其 `setenv.bash`。

### 必须手动设置的变量

```bash
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_HOME_PATH=$ASCEND_HOME/cann-8.5.1
export PATH=$ASCEND_HOME_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/aarch64-linux/lib64:$ASCEND_HOME/driver/lib64/driver:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

### `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 的作用

这是 PyTorch 2.9 引入的后端自动加载机制：默认情况下 torch 会扫描已安装的 `torch_*` 包并自动注册其 device。设为 `0` 禁用该行为，让用户显式 `import torch_npu` 后再注册 NPU 后端。

原因：若不禁用，当 `import torch_npu` 因 `libhccl.so`/`libascend_hal.so` 等环境变量缺失而失败时，torch 仍会自动抢占 `PrivateUse1` dispatch key，导致后续 `torch.ops.openjiuwen.*` 调用路由到错误后端。`__init__.py` 在 `import torch_npu` 失败时会抛出友好错误，提示补全 `ASCEND_HOME_PATH`/`LD_LIBRARY_PATH`。

## Python 依赖

`setup.py` 声明：

```python
install_requires = ["torch>=2.9.0", "torch-npu>=2.9.0"]
setup_requires = ["setuptools>=61.0", "wheel"]
```

`pyproject.toml` 仅声明 build-system，无项目元数据，真实元数据来自 `setup.py`。

安装构建依赖：

```bash
pip install torch==2.9.0 torch-npu==2.9.0 setuptools>=61.0 wheel
```

## torch_npu 与 CANN 的耦合

`setup.py` 在 `sys.path` 中搜索 `torch_npu/include`，因为 adapter 代码 include 了 torch_npu 内部头：

```cpp
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/npu/Module.h>
```

运行时链接 `-ltorch_npu`。因此 torch_npu 的 ABI 必须与 torch 2.9.0 对齐，且其 C++ 头与 CANN 8.5.1 的 `acl/acl_rt.h` ABI 兼容。

## 验证环境

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -c "import torch, torch_npu; print(torch.__version__, torch.npu.is_available())"
```

如果 `torch.npu.is_available()` 返回 False，通常是 `TORCH_DEVICE_BACKEND_AUTOLOAD` 未设为 0 或 `LD_LIBRARY_PATH` 缺少 driver/lib64。

下一步：[构建指南](01-build-guide.md)。
