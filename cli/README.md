# openjiuwen-plugin

openJiuwen Plugin CLI utilities（PyPI 发行名 **`openjiuwen-plugin`**；Python 中 **`import openjiuwen_plugin`**）。

## 命令入口（PyPI）

安装后控制台命令为 **`openjiuwen-plugin`**（与 `pip install` 名称一致；`pyproject.toml` 中 `[project.scripts]` 注册）。子命令直接跟在可执行文件后，例如 `openjiuwen-plugin init ...`。

## CLI 代码分层（当前实现）

可安装包在 **`openjiuwen_plugin/`**（与 `pyproject.toml` 同级，扁平布局）：

- `openjiuwen_plugin/main.py`：入口层，仅做日志初始化、参数解析、命令分发
- `openjiuwen_plugin/parsers.py`：参数层，集中维护各子命令 `argparse` 定义
- `openjiuwen_plugin/handlers.py`：处理层，集中维护各子命令执行逻辑与返回码
- `openjiuwen_plugin/plugin.py`：插件生命周期能力（init/validate/pack/publish/install）
- `openjiuwen_plugin/market.py`：marketplace HTTP 契约调用
- `openjiuwen_plugin/schemas/`：CLI 侧契约对象（与 marketplace schema 对齐的轻量模型）

### 契约分层约定（推荐）

- `market.py`：只负责 HTTP 调用、响应解析，返回 `schemas` 契约对象
- `handlers.py`：只负责命令输出（`logger`）与退出码，不拼装底层响应结构
- `schemas/__init__.py`：统一导出契约对象，调用方使用 `from openjiuwen_plugin.schemas import ...`
- 新增/变更市场接口时，建议先更新 `schemas`，再接入 `market` 与 `handlers`

## Implemented Commands

- `openjiuwen-plugin init <name>` — 生成插件脚手架（`--type`: tools | mcp-stdio | restful-api；`--path` 父目录默认当前目录）
- `openjiuwen-plugin validate <path>` — 校验插件目录与元数据（path 必填）
- `openjiuwen-plugin pack <path>` — 按类型打包（path 必填；`-o` 默认 `out`）
- `openjiuwen-plugin publish [path]` — 上传市场：无 `--file` 时 path 必填；`--file` 指定已有 zip 直接上传
- `openjiuwen-plugin info <asset_id> --version <v>` — 获取版本详情：GET `/api/v1/plugins/{asset_id}/versions/{v}`（公开接口，不鉴权），展示 `PluginVersionDetail` 主要字段（asset/plugin/publisher/changelog/file/icon）
- `openjiuwen-plugin search [query]` — 列表/搜索（见下方 **search 参数**）
- `openjiuwen-plugin delete <plugin_id>` — `DELETE /api/v1/plugins/{asset_id}/versions/{version}`；Bearer 需 `--user-id`；或 `--system-token` / `OPENJIUWEN_SYSTEM_TOKEN`
- `openjiuwen-plugin install <asset_id>` — `GET /api/v1/artifacts/{asset_id}` 拉取 zip（不鉴权），校验后按 `runtime.type` 执行 `pip install`（tools 优先 `dist/*.whl`）

> 说明：`init` 仅生成插件模板与基础配置；`install` 仅安装包和依赖，不会自动启动服务/进程。`mcp-stdio` 需手动（或由宿主运行器）执行 `mcp.command` 启动；`restful-api` 需用户实现并启动自己的 API 服务，再将 `api.base_url` 指向该服务。

## Install

在 **cli 目录**下执行（或从 PyPI `pip install openjiuwen-plugin` 后直接使用 `openjiuwen-plugin`）：

```bash
cd cli
pip install -e .   # 可编辑安装；或 pip install .
```

## 环境变量与配置（安装后 token / market-url 放哪）

CLI **不会**自动读取 `.env` 文件；需在你本机或 CI 的**进程环境**里配置（与多数命令行工具一致）。

| 变量 | 用途 | 典型场景 |
|------|------|----------|
| `OPENJIUWEN_MARKET_URL` | 市场基础 URL（不含 `/api/v1/...` 路径） | 省去每次 `--market-url` |
| `OPENJIUWEN_USER_TOKEN` | 发布/删除用用户 Bearer token | `publish` / `delete`（用户模式）：优先 `--token`，否则读此变量 |
| `OPENJIUWEN_USER_ID` | 发布/删除时的发布者 `user_id` | **`publish`** / **`delete`（Bearer）**：优先 `--user-id`，否则读此变量 |
| `OPENJIUWEN_SYSTEM_TOKEN` | 系统管理员 token（`X-System-Token`） | `publish` / `delete`（System 模式）：优先 `--system-token`，否则读此变量 |
| `LOG_LEVEL` | 日志级别，默认 `INFO` | 调试时可设 `DEBUG` |

**配置方式示例：**

- **Windows（当前用户，PowerShell 持久）**：`[Environment]::SetEnvironmentVariable("OPENJIUWEN_MARKET_URL", "http://127.0.0.1:8100", "User")`，新开终端生效。
- **Windows（仅当前会话）**：`$env:OPENJIUWEN_MARKET_URL="http://127.0.0.1:8100"`。
- **Linux / macOS**：在 `~/.bashrc`、`~/.zshrc` 等中加入 `export OPENJIUWEN_MARKET_URL=...`、`export OPENJIUWEN_USER_TOKEN=...`，`source` 后生效。
- **CI（GitHub Actions 等）**：在仓库 Secrets / Variables 里配置，流水线中 `env:` 注入。
- **IDE 终端**：在运行配置的 Environment 里填写上述变量。

> **安全**：不要把 token 写进仓库。**仅 `publish` / `delete` 使用 token**（`--token` / `OPENJIUWEN_USER_TOKEN`（Bearer）或 `--system-token` / `OPENJIUWEN_SYSTEM_TOKEN`（System））。`search`、`info` 与 **`install` 下载**为公开接口，**从不**携带鉴权头。

## search 支持的参数

调用 Store 的 `GET /api/v1/plugins`（`PluginListQuery`），**不携带 Authorization**。

| 选项 | 说明 |
|------|------|
| `[query]` | `search_keyword` |
| `--type` | `plugin_type`（精确匹配） |
| `--author NAME` | `publisher_name`（发布者展示名模糊；CLI 使用 `--author` 以兼容习惯） |
| `--asset-id ID` | `asset_id` |
| `--asset-type TYPE` | `asset_type` |
| `--publisher-id ID` | `publisher_id` |
| `--page N` | `page`（默认 1） |
| `--page-size N` | `page_size`（默认 20，最大 100） |
| `--order-by FIELD` | `order_by`：`install_count` \| `like_count` \| `create_time` \| `update_time` \| `review_count`（默认 `install_count`） |
| `--desc BOOL` | 是否降序（对应 API `desc`）；取值 `true/false`，默认 `true` |
| `--market-url` | 市场基础 URL（也可用 `OPENJIUWEN_MARKET_URL`） |

以上与 marketplace `PluginListQuery` 字段一致；**无** `offset` 等 API 不存在的参数。

示例：

```bash
openjiuwen-plugin search weather --type tools --page-size 20 --order-by create_time --market-url http://127.0.0.1:8100
openjiuwen-plugin search weather --type tools --order-by create_time --desc true --market-url http://127.0.0.1:8100
openjiuwen-plugin search weather --type tools --order-by create_time --desc false --market-url http://127.0.0.1:8100
```

## 文档索引

| 文档 | 内容 |
|------|------|
| 本文（`cli/README.md`） | 命令、环境变量、插件规范摘要 |
| 仓库根目录 **`CLI_FEATURE_DESIGN.md`** | 与 Store 对接的契约、鉴权、各子命令与 API 对应关系（设计稿） |
| `openjiuwen_plugin/parsers.py` 等 | CLI 参数构建、命令处理、入口分发的实现分层 |
| 仓库根目录 **`VERIFICATION.md`** | 本地联调逐步验证（marketplace / MinIO / 鉴权、`search` / `info` / `delete`、数据落库） |

## 本地联调验证（marketplace / MinIO / 鉴权）

以 **`VERIFICATION.md`** 为准；其中 marketplace 端口须与 `.env` 中 **`STORE_PORT`** 一致（仓库 **`.env.example`** 默认为 **8100**）。

## Tests

包名为 **`openjiuwen_plugin`**，测试在 **`cli/tests/`**。任选其一即可。

**方式 A：在 `cli` 目录下**（把 `cli` 当作包搜索根，与 `pip install -e .` 一致）：

```bash
cd cli
pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
```

未做 editable 安装时，临时指定 `PYTHONPATH`：

```powershell
cd cli
$env:PYTHONPATH = "."
python -m unittest discover -s tests -p "test_*.py" -v
```

```bash
cd cli
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py" -v
```

**方式 B：在仓库根目录 `agent-tools` 下**（仍支持；需让 `import openjiuwen_plugin` 能找到 **`cli/openjiuwen_plugin`**）：

```powershell
# 在 agent-tools 根目录
$env:PYTHONPATH = "cli"
python -m unittest discover -s cli/tests -p "test_*.py" -v
```

```bash
# 在 agent-tools 根目录
PYTHONPATH=cli python -m unittest discover -s cli/tests -p "test_*.py" -v
```

使用 **pytest** 时：在 **`cli`** 下执行且 `pyproject.toml` 已配置 `pythonpath = ["."]`；若在 **`agent-tools` 根** 跑 pytest，需等价加上 `PYTHONPATH=cli`（或 `pytest cli/tests --pythonpath=cli`，视本地 pytest 版本而定）。

## Usage

```bash
openjiuwen-plugin init demo-weather
openjiuwen-plugin init demo-mcp --type mcp-stdio
openjiuwen-plugin init my-api --type restful-api
openjiuwen-plugin validate ./demo-weather
openjiuwen-plugin pack ./demo-weather
openjiuwen-plugin info <发布返回的plugin_id> --version 1.0.0 --market-url http://127.0.0.1:8100

openjiuwen-plugin publish ./demo-weather --user-id user_001 --token <your_token> --market-url http://127.0.0.1:8100
openjiuwen-plugin publish --file ./out/demo-weather-1.0.0.zip --user-id user_001 --token <your_token> --market-url http://127.0.0.1:8100
openjiuwen-plugin publish ./demo-weather --user-id user_001 --system-token <your_system_token> --market-url http://127.0.0.1:8100

openjiuwen-plugin delete <asset_id> --system-token <your_system_token> --market-url http://127.0.0.1:8100
openjiuwen-plugin delete <asset_id> --version all --token <your_token> --user-id user_001 --market-url http://127.0.0.1:8100
```

### 打包与发布

1. **路径约定**：`validate`、`pack` 的 path 必填；`publish` 无 `--file` 时 path 必填。`info` 为 **asset_id（plugin_id） + `--version`** + market URL。
2. **打包**：`openjiuwen-plugin pack <path>`，`-o` 默认 `out/`。
3. **发布**：`--user-id` 与 `OPENJIUWEN_USER_ID` 二选一至少其一（命令行优先）；`--token/OPENJIUWEN_USER_TOKEN`（用户 Bearer）与 `--system-token/OPENJIUWEN_SYSTEM_TOKEN`（System）二选一；若选择用户 Bearer 且无 token 则交互输入；market 可用 `OPENJIUWEN_MARKET_URL`。**获取 plugin_id**：首版发布输出，或 `openjiuwen-plugin search <关键词>`。

## Generated Plugin Structure

`openjiuwen-plugin init <plugin-name>` generates:

```text
plugin-name/
  plugin.yaml
  README.md
  icon.png
  schemas/
    tools.json
  src/
    plugin_name/
      __init__.py
      plugin.py
```

`plugin_name` is derived from `plugin-name` by replacing `-` with `_`.

**mcp-stdio** (`openjiuwen-plugin init demo-mcp --type mcp-stdio`) 默认目录：

```text
demo-mcp/
  plugin.yaml
  README.md
  icon.png
  schemas/
    tools.json
  src/
    demo_mcp/
      __init__.py
      mcp_server.py
```

`plugin.yaml` 中 `runtime.type` 为 `mcp-stdio`，`mcp.command` 为 `["python", "-m", "demo_mcp.mcp_server"]`。`mcp_server.py` 使用 [FastMCP](https://github.com/PrefectHQ/fastmcp) 模板；`openjiuwen-plugin init` 会把 `fastmcp` 写入生成的 `pyproject.toml` 依赖。注意：`pip install .` / `openjiuwen-plugin install` 只做安装，不会自动调用 `mcp.run()`；需要手动或由宿主运行器执行 `mcp.command` 启动服务。

**restful-api**：scaffold 为 `rest_api.py` 与 `plugin.yaml` 中的 `api.base_url`，同时会生成 `schemas/tools.json`（仅模板占位，不参与该类型 validate 的工具校验）和 `pyproject.toml` 以支持 `pip install .`。注意：`rest_api.py` 当前为占位模板，需要你补全实际 REST 实现逻辑并手动启动服务（或部署到外部）。`install` 不会自动启动该服务。

## Plugin Spec (MVP)

This section defines the `plugin.yaml` and `schemas/tools.json` formats used by `openjiuwen-plugin init` and `openjiuwen-plugin validate`.

### `plugin.yaml`

- `name`: `^[a-z][a-z0-9-]*$`
- `version`: semver
- `display_name`, `description`: non-empty string
- `runtime.type`: `tools` | `mcp-stdio` | `restful-api`
- `metadata.author`, `metadata.tags`: required
- `compatibility.python`: **必填**，**PEP 440** 版本说明符字符串（与 `pyproject.toml` 的 `requires-python` 同类）：可写**一条**约束（如 `>=3.11`、`==3.11.*`、`~=3.11`）或**逗号分隔多条**表示区间（如 `>=3.11, <3.14`）；不能写裸版本号（如 `3.11`，须带比较符）
- `tools_schema`: required for `tools`, path `schemas/tools.json`
- `mcp`: required for `mcp-stdio` (transport, command)
- `api`: required for `restful-api` (base_url)

### `schemas/tools.json`

- `tools`: non-empty array; each tool has `name`, `description`, `input_schema`, `output_schema` (JSON Schema object type).

### Runtime Consistency

- **tools**: Tool names in `@tool(name="...")` in `plugin.py` must match `schemas/tools.json` (one-to-one).
- **mcp-stdio**: `mcp_server.py` must exist; `mcp.transport` = `stdio`, `mcp.command` non-empty array.
- **restful-api**: `rest_api.py` must exist; `api.base_url` non-empty string.
