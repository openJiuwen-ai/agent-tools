# openjiuwen-plugin（插件命令行工具）

面向 openJiuwen 插件市场的命令行工具：在本地生成与校验插件工程、打包上传、检索与安装。**PyPI 发行名将使用 `openjiuwen-plugin`**（安装后入口命令同名）；Python 包目录名为 `openjiuwen_plugin`。

> **分发说明**：CLI **尚未发布到 PyPI**。请从本仓库 **`cli/`** 目录以可编辑方式安装（见下文）。后续发布 PyPI 后，可直接 `pip install openjiuwen-plugin` 使用。

---

## 1. 环境要求

| 项目 | 说明 |
|------|------|
| Python | **≥ 3.11.4**（与 `pyproject.toml` 中 `requires-python` 一致） |
| 操作系统 | Windows / Linux / macOS 均可 |
| 对接市场 | 可选。使用市场相关子命令时需已知**市场根 URL**（见第 5 节；参数见第 6 节） |

---

## 2. 安装

在克隆后的仓库中进入 **`cli`** 目录执行：

```bash
cd cli
pip install -e .
```

- **`-e`（可编辑）**：修改源码后立即生效，适合开发与联调。
- 安装成功后，终端中应可使用命令 **`openjiuwen-plugin`**。建议执行 **`openjiuwen-plugin -h`** 确认子命令列表。

若安装成功但提示 **`openjiuwen-plugin` 不是内部或外部命令**（常见于 Windows 用户级安装路径未加入 PATH），可使用模块方式调用（效果相同）：

```bash
cd cli
python -m openjiuwen_plugin.main --help
```

---

## 3. 快速上手（不依赖市场）

以下命令仅需本地文件系统，**无需**访问市场服务。

```bash
# 1）生成 tools 类型脚手架（默认）
openjiuwen-plugin init demo-weather --path .

# 其他类型示例
openjiuwen-plugin init demo-mcp --path . --type mcp-stdio
openjiuwen-plugin init demo-api --path . --type restful-api
openjiuwen-plugin init my-skill --path . --type skill

# 2）校验插件目录
openjiuwen-plugin validate ./demo-weather

# 3）打包（默认输出到插件目录下的 out/）
openjiuwen-plugin pack ./demo-weather
# 或指定输出目录
openjiuwen-plugin pack ./demo-weather -o ./dist-zips
```

说明：

- **`tools`** 类型打包前会执行 `pip wheel`，请保证当前环境可访问 PyPI 或已配置 wheel 构建所需依赖。
- **`skill`** 类型默认**不生成**根目录 `README.md`；若自行添加，`pack` 会将其一并打入 zip。

---

## 4. 环境变量（仅系统 / 进程环境）

CLI **只**读取当前进程可见的环境变量（操作系统用户环境、系统环境、shell 中 `export` / `$env:`、CI 注入等），实现上为 Python **`os.environ`**。**不会**解析、加载仓库或任意路径下的 `.env` 文件；与 marketplace / Docker Compose 使用的 `.env` **无联动**。

若你在本机用 `.env` 启动市场服务，仍需在运行 CLI 的同一终端里**单独**设置下表变量（或写入用户级环境变量），例如从 `.env` 复制值后手动 `export`，或使用你习惯的 shell 插件在启动 CLI 前注入。

| 变量 | 作用 |
|------|------|
| `OPENJIUWEN_MARKET_URL` | 市场服务**根地址**（不含 `/api/v1/...`），例如 `http://127.0.0.1:8100` |
| `OPENJIUWEN_USER_TOKEN` | 用户 Bearer Token；用于 `publish` / `delete`（与 `--token` 二选一优先级：命令行优先） |
| `OPENJIUWEN_SYSTEM_TOKEN` | 系统管理员 Token（请求头 `X-System-Token`）；与 `--system-token` 同理 |
| `LOG_LEVEL` | 日志级别，默认 `INFO`；调试可设为 `DEBUG` |

**安全提示**：勿将 Token 写入仓库。`search`、`info`、`install`（下载 zip）对市场侧为**公开接口**，CLI **不会**附带用户鉴权头。

**配置示例（当前 PowerShell 会话）**：

```powershell
$env:OPENJIUWEN_MARKET_URL = "http://127.0.0.1:8100"
$env:OPENJIUWEN_USER_TOKEN = "<你的 Token>"
```

**配置示例（Linux / macOS）**：

```bash
export OPENJIUWEN_MARKET_URL=http://127.0.0.1:8100
export OPENJIUWEN_USER_TOKEN="<你的 Token>"
```

---

## 5. 对接市场服务（简要）

将 **`OPENJIUWEN_MARKET_URL`** 设为插件市场**根地址**（不含 `/api/v1/...`）；或在各子命令上使用 **`--market-url`**（**命令行优先于环境变量**）。未设置且命令也未带 `--market-url` 时，依赖市场的子命令会报错退出。

**版本号约定**：`plugin.yaml` 中的 `version` 与 **`--plugin-version`** 均须为 **`x.y.z`** 三位非负整数段（如 `1.0.0`），**不支持** `1.0.0-rc1` 等预发布后缀。允许前缀 **`v`/`V`**，规范化后上传。

**帮助**：`openjiuwen-plugin -h` 列出子命令；`openjiuwen-plugin <子命令> -h` 与下表一致，以程序输出为准。

**各子命令的全部参数**见第 **6** 节。

---

## 6. 子命令与参数

### 6.1 `init` — 生成脚手架

| 参数 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 插件名，如 `weather-plugin`（须满足 `^[a-z][a-z0-9-]*$`；`skill` 另有 Agent Skills 风格限制） |
| `--path` | 否 | 在哪个**父目录**下创建 `name/` 子目录，默认当前目录 `.` |
| `--type` | 否 | `tools`（默认）\| `mcp-stdio` \| `restful-api` \| `skill` |
| `--force` | 否 | 目标 `name` 目录已存在且非空时仍覆盖初始化 |

### 6.2 `validate` — 校验插件目录

| 参数 | 必填 | 说明 |
|------|------|------|
| `path` | 是 | 插件**根目录**（内含 `plugin.yaml` 等） |

### 6.3 `pack` — 打包为 zip

| 参数 | 必填 | 说明 |
|------|------|------|
| `path` | 是 | 插件根目录 |
| `-o` / `--output` | 否 | zip 输出目录，默认 `out`。若为**相对路径**，则相对于**插件根目录** `path` 解析；**绝对路径**则直接使用 |

打包前会先执行与 `validate` 相同的校验。`tools` 会执行 `pip wheel`（需网络或本地构建环境）。

### 6.4 `publish` — 上传市场

| 参数 | 必填 | 说明 |
|------|------|------|
| `path` | 条件 | 插件根目录；**与 `--file` 二选一**：未使用 `--file` 时**必填**（先 `pack` 再上传） |
| `-f` / `--file` | 条件 | 已有 zip 路径；指定则**跳过 pack**，直接上传该文件 |
| `--token` | 条件 | 普通用户：`Authorization: Bearer …`。与 `--system-token` **互斥**；可省略并从 `OPENJIUWEN_USER_TOKEN` 读取（命令行优先） |
| `--system-token` | 条件 | 系统管理员：`X-System-Token`。与 `--token` **互斥**；可省略并从 `OPENJIUWEN_SYSTEM_TOKEN` 读取 |
| `--market-url` | 条件 | 市场根 URL；未设则用 `OPENJIUWEN_MARKET_URL` |
| `--plugin-id` | 否 | 插件 ID；首发可省略，后续发版建议带上 |
| `--plugin-version` | 否 | 覆盖版本号（`x.y.z`，可选 `v` 前缀） |
| `--version-desc` | 否 | 版本说明 |
| `--force` | 否 | 允许覆盖市场侧已存在的同版本（以服务端策略为准） |

`publish`、`delete` 的用户或系统鉴权须通过 `--token` 或 `OPENJIUWEN_USER_TOKEN`（或 `--system-token` / `OPENJIUWEN_SYSTEM_TOKEN`）；未提供则报错退出。

### 6.5 `info` — 查询版本详情

不携带用户鉴权；需市场 URL。

| 参数 | 必填 | 说明 |
|------|------|------|
| `asset_id` | 是 | 资产 ID（与发布返回的 `plugin_id` 一致） |
| `-v` / `--version` | 是 | 目标版本，如 `1.0.0` |
| `--market-url` | 条件 | 市场根 URL；未设则用环境变量 |

### 6.6 `search` — 列表 / 搜索

不携带 Authorization。

| 参数 | 必填 | 说明 |
|------|------|------|
| `query` | 否 | 搜索关键词；可省略，表示空关键词 |
| `--market-url` | 条件 | 市场根 URL |
| `--type` | 否 | `plugin_type` 精确匹配：`tools` / `mcp-stdio` / `restful-api` / `skill` |
| `--author` | 否 | 发布者展示名（模糊） |
| `--asset-id` | 否 | 资产 ID |
| `--asset-type` | 否 | 资产类型 |
| `--publisher-id` | 否 | 发布者 ID |
| `--page` | 否 | 页码，默认 `1` |
| `--page-size` | 否 | 每页条数，默认 `20`，最大 `100` |
| `--order-by` | 否 | `install_count`（默认）\| `like_count` \| `create_time` \| `update_time` \| `review_count` |
| `--desc` | 否 | 是否降序：`true` / `false` / `yes` / `no` 等，默认 `true` |

### 6.7 `delete` — 删除版本或整包

| 参数 | 必填 | 说明 |
|------|------|------|
| `plugin_id` | 是 | 资产 ID |
| `--market-url` | 条件 | 市场根 URL |
| `--token` | 条件 | 普通用户 Bearer；与 `--system-token` **互斥**；可配合 `OPENJIUWEN_USER_TOKEN`（命令行优先） |
| `--system-token` | 条件 | 系统管理员 `X-System-Token`；与 `--token` **互斥**；可配合 `OPENJIUWEN_SYSTEM_TOKEN` |
| `--version` | 否 | 要删的版本号；**省略**则删除**全部**版本（与 CLI `--help` 描述一致；服务端行为以市场 API 为准） |

### 6.8 `install` — 下载并安装

不携带用户鉴权。按 zip 内 `runtime.type`：`skill` 仅复制技能目录，**不**执行 `pip`；其余类型会 `pip install`（`tools` 装 `dist/*.whl`，`mcp-stdio` / `restful-api` 为 `pip install .`）。**不会**自动启动 MCP/REST 服务。

| 参数 | 必填 | 说明 |
|------|------|------|
| `asset_id` | 是 | 市场 `asset_id` |
| `--market-url` | 条件 | 市场根 URL |
| `--version` | 否 | 保留参数；当前按 `asset_id` 拉取制品，**可能被忽略**（以命令行提示为准） |
| `--prefix` | 否 | 传给 `pip install --prefix`；**`skill` 类型忽略** |
| `-o` / `--output` | 否 | 解压与安装根目录，默认**当前工作目录** |
| `--save-zip` | 否 | 将下载的 zip **额外**保存到该路径 |
| `--force` | 否 | 目标目录已存在时允许覆盖 |

### 6.9 市场相关命令示例

```bash
BASE=http://127.0.0.1:8100   # 占位；或已 export OPENJIUWEN_MARKET_URL

openjiuwen-plugin publish ./demo-weather --token <TOKEN> --market-url $BASE
openjiuwen-plugin publish -f ./out/demo-weather-0.0.1.zip --token <TOKEN> --market-url $BASE
openjiuwen-plugin info <asset_id> -v 1.0.0 --market-url $BASE
openjiuwen-plugin search weather --type tools --page-size 20 --order-by create_time --market-url $BASE
openjiuwen-plugin install <asset_id> --market-url $BASE
openjiuwen-plugin install <asset_id> -o ./plugins --save-zip ./cache/p.zip --market-url $BASE
openjiuwen-plugin delete <asset_id> --version all --token <TOKEN> --market-url $BASE
```

（Windows PowerShell 可将 `$BASE` 换为变量或字面 URL。）

---

## 7. 代码结构（维护与二次开发）

可安装包位于 **`cli/openjiuwen_plugin/`**：

| 模块 | 职责 |
|------|------|
| `main.py` | 入口：日志初始化、解析参数、分发子命令 |
| `parsers.py` | 各子命令 `argparse` 定义 |
| `handlers.py` | 子命令业务编排、日志输出、进程退出码 |
| `plugin.py` | `init` / `validate` / `pack` / `publish` / `install` 等核心逻辑 |
| `market.py` | 对市场 HTTP API 的调用与响应解析 |
| `schemas/` | 与市场对齐的请求/响应模型（Pydantic） |
| `logging_config.py` | 控制台日志与敏感信息脱敏 |

**HTTP 与重试**：`publish`（上传）与 `delete` 为**单次请求、不自动重试**，避免非幂等写操作在「服务端已成功、客户端未收到响应」时重复提交。`search`、`info`、制品元数据与 zip 下载等**只读**调用在 `market.py` 内对网络错误及部分可恢复状态码做**有限次数退避重试**（细节见源码）。

约定：`market.py` 只负责 HTTP 与契约对象；`handlers.py` 负责面向用户的日志与返回码；扩展接口时优先改 **`schemas`**，再改 **`market`** 与 **`handlers`**。

---

## 8. 脚手架类型与目录结构

`init --type` 支持 **`tools`**（默认）、**`mcp-stdio`**、**`restful-api`**、**`skill`**。下图为各类型生成后的典型布局（包目录名由插件名将 `-` 换为 `_` 得到，如 `demo-mcp` → `demo_mcp`）。

### `tools`

```text
plugin-name/
  plugin.yaml
  README.md
  icon.png
  pyproject.toml
  schemas/tools.json
  src/plugin_name/
    __init__.py
    plugin.py
```

### `mcp-stdio`

```text
demo-mcp/
  plugin.yaml
  README.md
  icon.png
  pyproject.toml
  schemas/tools.json
  src/demo_mcp/
    __init__.py
    mcp_server.py
```

`mcp_server.py` 为 FastMCP 模板；依赖见生成的 `pyproject.toml`。

### `restful-api`

```text
demo-api/
  plugin.yaml
  README.md
  icon.png
  pyproject.toml
  schemas/tools.json
  src/demo_api/
    __init__.py
    rest_api.py
```

`rest_api.py` 为**可选占位**（便于本地写客户端或示例）；**服务已独立部署时**，宿主一般只需 **`plugin.yaml` 中的 `api.base_url`** 与 **`schemas/tools.json`** 即可知道如何调用，可不依赖 `rest_api.py`。  
**建议**在 `schemas/tools.json` 中用与 `tools` 类型相同的习惯描述各 REST 能力的入参、出参；当前 **`validate` 不对 `restful-api` 校验该文件**。

### `skill`

```text
my-skill/
  plugin.yaml
  icon.png
  my-skill/
    SKILL.md
    scripts/
    references/
    assets/
```

### 关于 `schemas/tools.json`

**`tools`、`mcp-stdio`、`restful-api`** 脚手架都会生成 **`schemas/tools.json`**（模板内容）。**`validate` 仅对 `runtime.type: tools` 校验该文件**（结构、与 `src/.../plugin.py` 中 `@tool(name=...)` 名称一致等）。**`mcp-stdio` 与 `restful-api`** 类型下该文件**不参与**上述校验；其中 **`restful-api` 仍建议在 `tools.json` 中维护各 API 能力的输入/输出说明**（结构可与 `tools` 类型对齐），只是 CLI 当前不强制校验。

---

## 9. `plugin.yaml` 与校验摘要（MVP）

| 字段 / 规则 | 说明 |
|-------------|------|
| `name` | 正则 `^[a-z][a-z0-9-]*$` |
| `version` | 与市场约定一致：**仅** `x.y.z` 三位数字段（如 `1.0.0`），无预发布/构建后缀 |
| `display_name`、`description` | 非空字符串 |
| `runtime.type` | `tools` \| `mcp-stdio` \| `restful-api` \| `skill` |
| `metadata.author`、`metadata.tags` | 必填；`tags` 为非空字符串数组 |
| `compatibility.python` | **非 skill 必填**；PEP 440 版本说明符（如 `>=3.11, <3.14`）；**skill 不要求** |
| `tools` 类型 | `tools_schema` 路径须为 `schemas/tools.json` 且文件存在 |
| `mcp-stdio` | `mcp.transport` 为 `stdio`，`mcp.command` 为非空字符串数组 |
| `restful-api` | 须含非空 `api.base_url`（见下节 **`api` 标准字段**） |
| `skill` | 恰好一个非隐藏子目录含 `SKILL.md`；frontmatter `name`/`description` 等规则与 Agent Skills 一致 |

### `restful-api`：`api` 标准字段与运行方式

**契约重心（已部署的 HTTP 服务）**：**`api.base_url`** + **`schemas/tools.json`**。前者给出服务根地址，后者用 `tools[]` 描述每个可调能力（名称、说明、`input_schema` / `output_schema` 等），宿主即可编排请求路径、方法与参数，**不必**使用 `rest_api.py`。

**`api` 对象约定**（当前 CLI **`validate` 仅强制校验**带星号的字段；其余为推荐扩展，宿主可自行约定解析方式）：

| 字段 | 必填 | 说明 |
|------|------|------|
| **`base_url`** | **是** | 服务根 URL（不含具体 path 时可只写到版本前缀，与 `tools.json` 内约定一致即可） |
| `openapi_url` | 否 | OpenAPI 文档地址（JSON/YAML）；与 `tools.json` 可二选一或并存，由宿主决定优先级 |
| `auth` | 否 | 鉴权说明，建议为对象，例如：`type: none \| bearer \| api_key`；`api_key` 时可含 `header`、`in`（`header` / `query`）等 |
| `default_headers` | 否 | 默认 HTTP 头（如 `Accept: application/json`），map 结构 |
| `timeout_seconds` | 否 | 建议超时秒数（数字），供宿主参考 |

**`rest_api.py`**：脚手架中的占位文件，**可保留**；需要本地封装调用逻辑时再实现，**不是**「仅远程服务」场景的必需项。

### `schemas/tools.json`（校验范围）

当 **`runtime.type` 为 `tools`** 时：`tools` 数组非空；每项含 `name`、`description`、`input_schema`、`output_schema`，且 schema 根类型为 `object`；`src/.../plugin.py` 中 `@tool(name="...")` 与 JSON 中工具名须一一对应。

**`mcp-stdio` / `restful-api`** 虽可带同名文件，**不进行**上述校验（见第 8 节）。**`restful-api`** 仍**建议**用该文件描述 REST 能力的参数形态，与运行时实现保持一致。

---

## 10. 开发与测试

在 **`cli`** 目录下：

```bash
pip install -e .
pip install pytest
python -m pytest -q
```

未做可编辑安装时，可临时设置 `PYTHONPATH`：

```powershell
cd cli
$env:PYTHONPATH = "."
python -m pytest -q
```

---

## 11. 常见问题

**Q：`openjiuwen-plugin` 命令找不到？**  
A：使用 `python -m openjiuwen_plugin.main`，或检查 Python 的 `Scripts` 目录是否已加入 PATH。

**Q：提示连不上市场？**  
A：确认 `OPENJIUWEN_MARKET_URL` 为**根地址**（不要带 `/api/v1`），且该 URL 在本机或网络内可访问。

**Q：发布失败提示版本格式错误？**  
A：使用 `x.y.z` 三位数字版本；不要使用 `1.0.0-rc1` 等形式。

**Q：路径含空格报错？**  
A：在 shell 中为路径加引号，例如 `openjiuwen-plugin validate "D:\My Plugins\demo"`。

**Q：只执行 `openjiuwen-plugin` 不带子命令会怎样？**  
A：会打印总帮助并退出（退出码非 0）；查看子命令请用 `openjiuwen-plugin -h` 或 `openjiuwen-plugin <子命令> -h`。

**Q：`LOG_LEVEL=DEBUG` 有什么用？**  
A：日志更详细；部分底层步骤（如释放 HTTP 连接失败）仅在 DEBUG 下可见。
