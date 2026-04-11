import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from openjiuwen_plugin.main import main
from openjiuwen_plugin.market import (
    PublishError,
    plugin_info,
    plugin_install_download,
    plugin_search,
)
from openjiuwen_plugin.plugin import (
    plugin_describe_local,
    plugin_init,
    plugin_install,
    plugin_pack,
    plugin_validate,
)
from openjiuwen_plugin.schemas import (
    PluginListQuery,
    PluginVersionDetail,
    PublishPluginInput,
    PublishRequest,
    SkillImportItemResult,
    SkillImportResponse,
    SkillImportSummary,
)


class PluginCommandsTest(unittest.TestCase):
    def test_publish_plugin_input_strips_v_and_validates_plugin_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "p"
            root.mkdir()
            inp = PublishPluginInput(plugin_path=root, plugin_version=" v1.2.3 ")
            self.assertEqual(inp.plugin_version, "1.2.3")

    def test_publish_plugin_input_rejects_prerelease_plugin_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "p"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "marketplace format"):
                PublishPluginInput(plugin_path=root, plugin_version="1.0.0-rc1")

    def test_publish_request_invalid_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "demo.zip"
            zip_path.write_bytes(b"PK\x03\x04")
            with self.assertRaisesRegex(ValueError, "checksum_sha256"):
                PublishRequest(
                    zip_path=zip_path,
                    checksum_sha256="bad-checksum",
                )

    def test_publish_request_zip_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "missing.zip"
            with self.assertRaisesRegex(ValueError, "zip file not found"):
                PublishRequest(
                    zip_path=zip_path,
                    checksum_sha256="a" * 64,
                )

    def test_init_and_validate_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("demo-plugin", Path(tmp))
            self.assertTrue((plugin_root / "schemas" / "tools.json").exists())
            self.assertTrue((plugin_root / "src" / "demo_plugin" / "plugin.py").exists())
            result = plugin_validate(plugin_root)
            self.assertTrue(result.ok, msg=f"errors: {result.errors}")

    def test_init_and_validate_mcp_stdio_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("demo-mcp", Path(tmp), plugin_type="mcp-stdio")
            self.assertTrue((plugin_root / "schemas" / "tools.json").exists())
            self.assertTrue((plugin_root / "src" / "demo_mcp" / "mcp_server.py").exists())
            self.assertTrue((plugin_root / "pyproject.toml").exists())
            result = plugin_validate(plugin_root)
            self.assertTrue(result.ok, msg=f"errors: {result.errors}")

    def test_validate_fails_with_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "broken"
            root.mkdir(parents=True)
            (root / "plugin.yaml").write_text("name: broken-plugin\nversion: 0.1.0\n", encoding="utf-8")
            result = plugin_validate(root)
            self.assertFalse(result.ok)
            self.assertGreater(len(result.errors), 0)

    def test_validate_detects_tool_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("mismatch-plugin", Path(tmp))
            plugin_py = plugin_root / "src" / "mismatch_plugin" / "plugin.py"
            plugin_py.write_text(
                """from openjiuwen.core.foundation.tool import tool

@tool(name="another-tool", description="x", input_params={})
def another_tool() -> dict:
    return {"ok": True}
""",
                encoding="utf-8",
            )
            result = plugin_validate(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("another-tool" in e for e in result.errors))

    def test_validate_fails_with_invalid_compatibility_specifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("compat-plugin", Path(tmp))
            plugin_yaml = plugin_root / "plugin.yaml"
            content = plugin_yaml.read_text(encoding="utf-8")
            content = content.replace(">=3.11, <3.14", "3.11")
            plugin_yaml.write_text(content, encoding="utf-8")

            result = plugin_validate(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("compatibility.python" in e for e in result.errors))

    def test_validate_compatibility_extra_keys_not_validated(self) -> None:
        """仅校验 compatibility.python；其它键（如 openjiuwen）CLI 不校验格式。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("compat-extra", Path(tmp))
            plugin_yaml = plugin_root / "plugin.yaml"
            data = yaml.safe_load(plugin_yaml.read_text(encoding="utf-8"))
            assert isinstance(data, dict) and isinstance(data.get("compatibility"), dict)
            data["compatibility"]["openjiuwen"] = "latest"
            plugin_yaml.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            result = plugin_validate(plugin_root)
            self.assertTrue(result.ok, msg=f"errors: {result.errors}")

    def test_validate_rejects_prerelease_version(self) -> None:
        """与 marketplace 一致：version 仅允许 x.y.z 三位数字。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("ver-demo", Path(tmp), plugin_type="mcp-stdio")
            p = plugin_root / "plugin.yaml"
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            assert isinstance(data, dict)
            data["version"] = "1.0.0-rc1"
            p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            result = plugin_validate(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("x.y.z" in e for e in result.errors))

    def test_cli_init_via_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["init", "demo-plugin", "--path", tmp])
            self.assertEqual(code, 0)

    def test_cli_init_with_mcp_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["init", "demo-mcp", "--path", tmp, "--type", "mcp-stdio"])
            self.assertEqual(code, 0)

    def test_pack_success(self) -> None:
        """mcp-stdio 类型整目录打包，不依赖 wheel。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("pack-demo", Path(tmp), plugin_type="mcp-stdio")
            out_dir = Path(tmp) / "out"
            zip_path = plugin_pack(plugin_root, out_dir)
            self.assertTrue(zip_path.exists())
            self.assertEqual(zip_path.name, "pack-demo-0.0.1.zip")
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
            self.assertTrue(any("plugin.yaml" in n for n in names))
            self.assertTrue(any("pack-demo-0.0.1" in n for n in names))

    def test_pack_excludes_plugin_out_directory(self) -> None:
        """mcp/rest 整目录打包时不应包含插件根目录下的 out/（避免历史 zip 被打进新包）。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("skip-mcp", Path(tmp), plugin_type="mcp-stdio")
            (plugin_root / "out").mkdir(parents=True, exist_ok=True)
            (plugin_root / "out" / "stale.zip").write_bytes(b"dummy")
            zip_path = plugin_pack(plugin_root, Path(tmp) / "publish-out")
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
            prefix = "skip-mcp-0.0.1"
            self.assertFalse(any(n.replace("\\", "/").startswith(f"{prefix}/out/") for n in names))

    def test_init_skill_validate_pack_install_no_pip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("demo-skill", Path(tmp), plugin_type="skill")
            self.assertTrue((plugin_root / "demo-skill" / "SKILL.md").is_file())
            self.assertTrue((plugin_root / "demo-skill" / "scripts").is_dir())
            self.assertFalse((plugin_root / "src").exists())
            self.assertFalse((plugin_root / "README.md").exists())
            result = plugin_validate(plugin_root)
            self.assertTrue(result.ok, msg=f"errors: {result.errors}")
            out_dir = Path(tmp) / "out"
            zip_path = plugin_pack(plugin_root, out_dir)
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
            prefix = "demo-skill-0.0.1"
            self.assertFalse(any("README.md" in n for n in names))
            self.assertFalse(any(f"{prefix}/demo-skill/scripts/" in n.replace("\\", "/") for n in names))
            inst = Path(tmp) / "install_root"
            with patch("openjiuwen_plugin.plugin.subprocess.run") as m_run:
                skill_dir = plugin_install(zip_path, extract_dir=inst)
            m_run.assert_not_called()
            self.assertTrue((skill_dir / "SKILL.md").is_file())
            self.assertFalse((inst / "demo-skill-0.1.0").exists())

    def test_init_restful_api_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("demo-api", Path(tmp), plugin_type="restful-api")
            self.assertTrue((plugin_root / "schemas" / "tools.json").exists())
            self.assertTrue((plugin_root / "src" / "demo_api" / "rest_api.py").exists())
            self.assertTrue((plugin_root / "pyproject.toml").exists())
            result = plugin_validate(plugin_root)
            self.assertTrue(result.ok, msg=f"errors: {result.errors}")

    def test_install_mcp_stdio_skips_pip_copies_bundle_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("demo-mcp", Path(tmp), plugin_type="mcp-stdio")
            out_dir = Path(tmp) / "out"
            zip_path = plugin_pack(plugin_root, out_dir)
            inst = Path(tmp) / "install_root"

            with patch("openjiuwen_plugin.plugin.subprocess.run") as m_run:
                m_run.return_value = None
                installed_root = plugin_install(zip_path, extract_dir=inst)

            self.assertTrue((installed_root / "plugin.yaml").exists())
            m_run.assert_not_called()

    def test_install_restful_api_skips_pip_copies_bundle_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("demo-api", Path(tmp), plugin_type="restful-api")
            out_dir = Path(tmp) / "out"
            zip_path = plugin_pack(plugin_root, out_dir)
            inst = Path(tmp) / "install_root"

            with patch("openjiuwen_plugin.plugin.subprocess.run") as m_run:
                m_run.return_value = None
                installed_root = plugin_install(zip_path, extract_dir=inst)

            self.assertTrue((installed_root / "plugin.yaml").exists())
            m_run.assert_not_called()

    def test_install_tools_wheel_only_zip_calls_pip_install_whl(self) -> None:
        """tools 发布包仅含 dist/*.whl 时，install 应调用 pip 将 wheel 装入当前 Python 环境。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("demo-wheel", Path(tmp), plugin_type="tools")
            out_dir = Path(tmp) / "out"
            zip_path = plugin_pack(plugin_root, out_dir)
            inst = Path(tmp) / "install_root"

            with patch("openjiuwen_plugin.plugin.subprocess.run") as m_run:
                m_run.return_value = None
                installed_root = plugin_install(zip_path, extract_dir=inst)

            self.assertTrue((installed_root / "plugin.yaml").exists())
            self.assertEqual(m_run.call_count, 1)
            pip_cmd = m_run.call_args[0][0]
            self.assertIn("-m", pip_cmd)
            self.assertIn("pip", pip_cmd)
            i_install = pip_cmd.index("install")
            self.assertEqual(pip_cmd[i_install + 1], "--")
            self.assertTrue(any(str(x).endswith(".whl") for x in pip_cmd))

    def test_info_reads_readme_local(self) -> None:
        """本地插件目录读取（plugin_describe_local），非 CLI 市场接口。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("info-demo", Path(tmp))
            info = plugin_describe_local(plugin_root)
            self.assertEqual(info.get("name"), "info-demo")
            self.assertEqual(info.get("version"), "0.0.1")
            self.assertIsNotNone(info.get("readme"))
            self.assertIn("info-demo", info["readme"])

    def test_info_from_market(self) -> None:
        """plugin info 通过版本详情 API 拉取摘要字段。"""
        with patch("openjiuwen_plugin.handlers.plugin_info") as m:
            m.return_value = PluginVersionDetail.model_validate(
                {
                    "asset_id": "demo-id",
                    "version": "1.0.0",
                    "asset_type": "plugin",
                    "plugin_type": "tools",
                    "name": "demo-plugin",
                    "display_name": "Demo Plugin",
                    "publisher_id": "u-1",
                    "publisher_name": "Alice",
                    "file_path": "plugins/u-1/demo-id/1.0.0/demo-plugin-1.0.0.zip",
                    "icon_uri": "http://example.com/icon.png",
                    "changelog": "initial release",
                }
            )
            code = main(["info", "demo-id", "--version", "1.0.0", "--market-url", "http://localhost:8000"])
            self.assertEqual(code, 0)
            m.assert_called_once()
            call_kw = m.call_args
            self.assertEqual(call_kw[0][1], "demo-id")
            self.assertEqual(call_kw[0][2], "1.0.0")

    def test_plugin_info_uses_versions_path(self) -> None:
        with patch("openjiuwen_plugin.market.requests.get") as m:
            m.return_value.status_code = 200
            m.return_value.headers = {"content-type": "application/json"}
            m.return_value.json.return_value = {
                "code": 200,
                "message": "ok",
                "data": {
                    "asset_id": "demo-id",
                    "version": "1.0.0",
                    "asset_type": "plugin",
                    "plugin_type": "tools",
                    "name": "demo-plugin",
                    "display_name": "Demo Plugin",
                    "publisher_id": "u-1",
                    "publisher_name": "Alice",
                },
            }
            detail = plugin_info("http://127.0.0.1:8100", "demo-id", "1.0.0")
            self.assertEqual(detail.asset_id, "demo-id")
            called_url = m.call_args[0][0]
            self.assertTrue(called_url.endswith("/api/v1/plugins/demo-id/versions/1.0.0"))

    def test_plugin_info_missing_fields_not_error(self) -> None:
        with patch("openjiuwen_plugin.market.requests.get") as m:
            m.return_value.status_code = 200
            m.return_value.headers = {"content-type": "application/json"}
            m.return_value.json.return_value = {
                "code": 200,
                "message": "ok",
                "data": {
                    "asset_id": "demo-id",
                    "version": "1.0.0",
                    # display_name / publisher_name 等缺失
                },
            }
            detail = plugin_info("http://127.0.0.1:8100", "demo-id", "1.0.0")
            self.assertEqual(detail.asset_id, "demo-id")
            self.assertEqual(detail.version, "1.0.0")
            self.assertEqual(detail.display_name, "")

    def test_publish_with_system_token(self) -> None:
        """publish 使用 --system-token 时应走 X-System-Token 路径。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("publish-sys-demo", Path(tmp))
            out_zip = Path(tmp) / "out"
            zip_path = plugin_pack(plugin_root, out_zip)

            # patch ``plugin.plugin_upload``（即 market 的 multipart 上传），避免真实网络请求。
            with patch.dict(os.environ, {"OPENJIUWEN_USER_TOKEN": ""}, clear=False):
                with patch("openjiuwen_plugin.plugin.plugin_upload") as m:
                    # 提供 --system-token；清空用户 token 环境以免与 system 双鉴权冲突
                    code = main(
                        [
                            "publish",
                            "--file",
                            str(zip_path),
                            "--system-token",
                            "sys-token-123",
                            "--market-url",
                            "http://localhost:8000",
                        ]
                    )
                    self.assertEqual(code, 0)

                    self.assertEqual(m.call_count, 1)
                    # plugin_upload(...) 来自 market 模块，经 plugin 命名空间绑定
                    call_args = m.call_args
                    self.assertEqual(call_args[0][0], "http://localhost:8000")
                    self.assertIsNone(call_args[0][1])
                    self.assertEqual(call_args[0][2], "sys-token-123")
                    self.assertIsInstance(call_args[0][3], PublishRequest)

    def test_delete_rejects_both_token_and_system_token(self) -> None:
        """delete 同时传 --token 与 --system-token 时应直接失败。"""
        with patch("openjiuwen_plugin.market.plugin_delete") as m:
            code = main(
                [
                    "delete",
                    "demo-id",
                    "--market-url",
                    "http://localhost:8000",
                    "--token",
                    "user-token-123",
                    "--system-token",
                    "sys-token-123",
                    "--version",
                    "all",
                ]
            )
            self.assertEqual(code, 1)
            m.assert_not_called()

    def test_pack_tools_wheel_zip(self) -> None:
        """tools 类型：先 build wheel，zip 含元数据 + dist/*.whl，不含 src/ 源码树。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("wheel-demo", Path(tmp))
            out_dir = Path(tmp) / "out"
            zip_path = plugin_pack(plugin_root, out_dir)
            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
            self.assertTrue(any("plugin.yaml" in n for n in names))
            self.assertTrue(any("schemas/tools.json" in n for n in names))
            self.assertTrue(any("README.md" in n for n in names))
            self.assertTrue(any("icon.png" in n for n in names))
            self.assertTrue(any("dist/" in n and n.endswith(".whl") for n in names))
            norm = [n.replace("\\", "/") for n in names]
            self.assertFalse(any("/src/" in n for n in norm))

    def test_pack_fails_when_plugin_yaml_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty"
            root.mkdir(parents=True)
            with self.assertRaises(ValueError) as ctx:
                plugin_pack(root)
            self.assertIn("plugin.yaml", str(ctx.exception))

    def test_pack_fails_when_validation_fails(self) -> None:
        """pack 会先执行校验，校验不通过则不打 zip。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "minimal"
            root.mkdir(parents=True)
            (root / "plugin.yaml").write_text(
                "name: minimal-pack\nversion: 0.1.0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                plugin_pack(root, Path(tmp) / "out")
            self.assertIn("validation failed", str(ctx.exception))

    def test_plugin_search_maps_plugin_list_query_params(self) -> None:
        """plugin_search 透传 PluginListQuery 相关 query 参数。"""
        with patch("openjiuwen_plugin.market.requests.get") as m:
            m.return_value.status_code = 200
            m.return_value.headers = {"content-type": "application/json"}
            m.return_value.json.return_value = {
                "code": 200,
                "message": "ok",
                "data": {"page": 3, "page_size": 15, "items": [], "total": 0},
            }
            plugin_search(
                "http://127.0.0.1:9",
                PluginListQuery(
                    search_keyword="keyword",
                    plugin_type="tools",
                    publisher_name="Alice",
                    asset_id="asset-1",
                    asset_type="plugin",
                    publisher_id="pub-1",
                    page=3,
                    page_size=15,
                    order_by="create_time",
                    desc=True,
                ),
            )
            m.assert_called_once()
            params = m.call_args[1]["params"]
            self.assertEqual(params["search_keyword"], "keyword")
            self.assertEqual(params["plugin_type"], "tools")
            self.assertEqual(params["publisher_name"], "Alice")
            self.assertEqual(params["asset_id"], "asset-1")
            self.assertEqual(params["asset_type"], "plugin")
            self.assertEqual(params["publisher_id"], "pub-1")
            self.assertEqual(params["page"], 3)
            self.assertEqual(params["page_size"], 15)
            self.assertEqual(params["order_by"], "create_time")
            self.assertTrue(params["desc"])

    def test_plugin_search_desc_true_by_default(self) -> None:
        with patch("openjiuwen_plugin.market.requests.get") as m:
            m.return_value.status_code = 200
            m.return_value.headers = {"content-type": "application/json"}
            m.return_value.json.return_value = {"code": 200, "message": "ok", "data": {"items": [], "total": 0}}
            plugin_search("http://127.0.0.1:9", PluginListQuery(order_by="install_count"))
            self.assertTrue(m.call_args[1]["params"]["desc"])

    def test_plugin_search_desc_true_when_flag_set(self) -> None:
        with patch("openjiuwen_plugin.market.requests.get") as m:
            m.return_value.status_code = 200
            m.return_value.headers = {"content-type": "application/json"}
            m.return_value.json.return_value = {"code": 200, "message": "ok", "data": {"items": [], "total": 0}}
            plugin_search("http://127.0.0.1:9", PluginListQuery(order_by="install_count", desc=True))
            self.assertTrue(m.call_args[1]["params"]["desc"])

    def test_plugin_search_desc_false_when_explicitly_set(self) -> None:
        with patch("openjiuwen_plugin.market.requests.get") as m:
            m.return_value.status_code = 200
            m.return_value.headers = {"content-type": "application/json"}
            m.return_value.json.return_value = {"code": 200, "message": "ok", "data": {"items": [], "total": 0}}
            plugin_search("http://127.0.0.1:9", PluginListQuery(order_by="install_count", desc=False))
            self.assertFalse(m.call_args[1]["params"]["desc"])

    def test_plugin_install_download_verifies_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plugin.zip"
            zip_bytes = (
                b"PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )
            import hashlib

            digest = hashlib.sha256(zip_bytes).hexdigest()
            meta_resp = SimpleNamespace(
                ok=True,
                status_code=200,
                headers={"content-type": "application/json"},
                json=lambda: {"data": {"download_url": "http://download.local/plugin.zip", "checksum_sha256": digest}},
                text="",
            )
            file_resp = SimpleNamespace(
                ok=True,
                status_code=200,
                headers={"content-type": "application/zip"},
                iter_content=lambda chunk_size=0: [zip_bytes],
                text="",
            )
            with patch("openjiuwen_plugin.market.requests.get", side_effect=[meta_resp, file_resp]) as m_get:
                info = plugin_install_download("http://market.local", "asset-1", out)
            self.assertTrue(out.exists())
            self.assertTrue(info.verified)
            self.assertEqual(info.actual_checksum_sha256, digest)
            self.assertEqual(m_get.call_count, 2)

    def test_plugin_install_download_checksum_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plugin.zip"
            zip_bytes = (
                b"PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )
            meta_resp = SimpleNamespace(
                ok=True,
                status_code=200,
                headers={"content-type": "application/json"},
                json=lambda: {
                    "data": {
                        "download_url": "http://download.local/plugin.zip",
                        "checksum_sha256": "0" * 64,
                    }
                },
                text="",
            )
            file_resp = SimpleNamespace(
                ok=True,
                status_code=200,
                headers={"content-type": "application/zip"},
                iter_content=lambda chunk_size=0: [zip_bytes],
                text="",
            )
            with patch("openjiuwen_plugin.market.requests.get", side_effect=[meta_resp, file_resp]):
                with self.assertRaises(RuntimeError) as ctx:
                    plugin_install_download("http://market.local", "asset-1", out)
            self.assertIn("checksum mismatch", str(ctx.exception))
            self.assertFalse(out.exists())

    def test_validate_rejects_omitted_runtime_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("omit-rt", Path(tmp))
            py = plugin_root / "plugin.yaml"
            data = yaml.safe_load(py.read_text(encoding="utf-8"))
            assert isinstance(data, dict) and isinstance(data.get("runtime"), dict)
            del data["runtime"]["type"]
            py.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            result = plugin_validate(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("runtime.type" in e for e in result.errors))

    def test_validate_rejects_unknown_runtime_type(self) -> None:
        """显式填写未知 runtime.type 须报错；未默认成 tools。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("rt-unknown", Path(tmp))
            py = plugin_root / "plugin.yaml"
            data = yaml.safe_load(py.read_text(encoding="utf-8"))
            assert isinstance(data, dict) and isinstance(data.get("runtime"), dict)
            data["runtime"]["type"] = "custom-unknown"
            py.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            result = plugin_validate(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("runtime.type" in e for e in result.errors))

    def test_validate_skill_fails_invalid_frontmatter_name_slug(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("slug-skill", Path(tmp), plugin_type="skill")
            skill_md = plugin_root / "slug-skill" / "SKILL.md"
            text = skill_md.read_text(encoding="utf-8").replace("name: slug-skill", "name: Bad_Name")
            skill_md.write_text(text, encoding="utf-8")
            result = plugin_validate(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("SKILL.md" in e or "name" in e for e in result.errors))

    def test_validate_skill_fails_when_frontmatter_name_differs_from_plugin_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("same-skill", Path(tmp), plugin_type="skill")
            skill_md = plugin_root / "same-skill" / "SKILL.md"
            text = skill_md.read_text(encoding="utf-8").replace("name: same-skill", "name: other-skill")
            skill_md.write_text(text, encoding="utf-8")
            result = plugin_validate(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("skill directory name" in e or "plugin.yaml name" in e for e in result.errors))

    def test_validate_skill_fails_empty_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("desc-skill", Path(tmp), plugin_type="skill")
            skill_md = plugin_root / "desc-skill" / "SKILL.md"
            text = skill_md.read_text(encoding="utf-8")
            text = text.replace(
                'description: "TODO: describe this skill for models and users"',
                'description: "   "',
            )
            skill_md.write_text(text, encoding="utf-8")
            result = plugin_validate(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("description" in e.lower() for e in result.errors))

    def test_install_second_install_without_force_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("twice-mcp", Path(tmp), plugin_type="mcp-stdio")
            zip_path = plugin_pack(plugin_root, Path(tmp) / "out")
            inst = Path(tmp) / "install_root"
            with patch("openjiuwen_plugin.plugin.subprocess.run") as m_run:
                m_run.return_value = None
                plugin_install(zip_path, extract_dir=inst)
                m_run.reset_mock()
                with self.assertRaises(FileExistsError):
                    plugin_install(zip_path, extract_dir=inst, force=False)

    def test_install_second_install_with_force_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = plugin_init("force-mcp", Path(tmp), plugin_type="mcp-stdio")
            zip_path = plugin_pack(plugin_root, Path(tmp) / "out")
            inst = Path(tmp) / "install_root"
            with patch("openjiuwen_plugin.plugin.subprocess.run") as m_run:
                m_run.return_value = None
                p1 = plugin_install(zip_path, extract_dir=inst)
                p2 = plugin_install(zip_path, extract_dir=inst, force=True)
            self.assertEqual(p1, p2)
            self.assertTrue((p2 / "plugin.yaml").is_file())

    def test_cli_init_skill_via_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["init", "cli-skill", "--path", tmp, "--type", "skill"])
            self.assertEqual(code, 0)
            root = Path(tmp) / "cli-skill"
            self.assertTrue((root / "cli-skill" / "SKILL.md").is_file())

    def test_init_rejects_unknown_plugin_type_from_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                plugin_init("x", Path(tmp), plugin_type="not-a-supported-type")
            self.assertIn("plugin type", str(ctx.exception).lower())

    def _skill_import_ok_response(self) -> SkillImportResponse:
        return SkillImportResponse(
            summary=SkillImportSummary(total=1, ok=1, failed=0),
            results=[
                SkillImportItemResult(
                    entry="skill-a",
                    status="ok",
                    plugin_id="pid-1",
                    name="skill-a",
                    version="0.0.1",
                )
            ],
        )

    def test_skill_import_cli_zip_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "bundle.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("skill-a/SKILL.md", b"# x")
            with patch(
                "openjiuwen_plugin.handlers.skill_import",
                return_value=self._skill_import_ok_response(),
            ) as m:
                code = main(
                    [
                        "skill-import",
                        str(zip_path),
                        "--market-url",
                        "http://localhost:8000",
                        "--system-token",
                        "sys-tok",
                    ]
                )
            self.assertEqual(code, 0)
            m.assert_called_once()
            kw = m.call_args.kwargs
            self.assertEqual(kw["zip_path"].resolve(), zip_path.resolve())
            self.assertFalse(kw["force"])
            self.assertFalse(kw["fail_fast"])

    def test_skill_import_cli_passes_force_and_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "bundle.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("x.txt", b"y")
            with patch(
                "openjiuwen_plugin.handlers.skill_import",
                return_value=self._skill_import_ok_response(),
            ) as m:
                code = main(
                    [
                        "skill-import",
                        str(zip_path),
                        "--market-url",
                        "http://localhost:8000",
                        "--system-token",
                        "t",
                        "--force",
                        "--fail-fast",
                    ]
                )
            self.assertEqual(code, 0)
            kw = m.call_args.kwargs
            self.assertTrue(kw["force"])
            self.assertTrue(kw["fail_fast"])

    def test_skill_import_cli_directory_packs_then_uploads_zip(self) -> None:
        """目录会先打成临时 zip 再上传；请求里的 zip_path 应为该临时文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "my-bundle"
            bundle_dir.mkdir()
            (bundle_dir / "a.txt").write_text("hi", encoding="utf-8")
            seen_zip: list[Path] = []

            def _upload(_market_url: str, _system_token: str, **kwargs):
                seen_zip.append(Path(kwargs["zip_path"]))
                return self._skill_import_ok_response()

            with patch("openjiuwen_plugin.handlers.skill_import", side_effect=_upload):
                code = main(
                    [
                        "skill-import",
                        str(bundle_dir),
                        "--market-url",
                        "http://localhost:8000",
                        "--system-token",
                        "t",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(len(seen_zip), 1)
            z = seen_zip[0]
            self.assertTrue(z.name.endswith(".zip"))
            self.assertNotEqual(z.resolve(), bundle_dir.resolve())
            # 打包目录在 finally 中删除后，临时 zip 不应仍存在
            self.assertFalse(z.is_file())

    def test_skill_import_requires_market_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "b.zip"
            zip_path.write_bytes(b"PK\x03\x04")
            with patch.dict(os.environ, {"OPENJIUWEN_MARKET_URL": ""}, clear=False):
                code = main(["skill-import", str(zip_path), "--system-token", "t"])
            self.assertEqual(code, 1)

    def test_skill_import_requires_system_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "b.zip"
            zip_path.write_bytes(b"PK\x03\x04")
            with patch.dict(os.environ, {"OPENJIUWEN_SYSTEM_TOKEN": ""}, clear=False):
                code = main(
                    ["skill-import", str(zip_path), "--market-url", "http://127.0.0.1:9"],
                )
            self.assertEqual(code, 1)

    def test_skill_import_bundle_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent.zip"
            code = main(
                [
                    "skill-import",
                    str(missing),
                    "--market-url",
                    "http://127.0.0.1:9",
                    "--system-token",
                    "t",
                ]
            )
        self.assertEqual(code, 1)

    def test_skill_import_publish_error_exits_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "bundle.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("x", b"y")
            with patch(
                "openjiuwen_plugin.handlers.skill_import",
                side_effect=PublishError(403, "forbidden"),
            ):
                code = main(
                    [
                        "skill-import",
                        str(zip_path),
                        "--market-url",
                        "http://localhost:8000",
                        "--system-token",
                        "t",
                    ]
                )
        self.assertEqual(code, 1)

    def test_skill_import_exits_1_when_summary_has_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "bundle.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("x", b"y")
            resp = SkillImportResponse(
                summary=SkillImportSummary(total=2, ok=1, failed=1),
                results=[
                    SkillImportItemResult(entry="a", status="ok"),
                    SkillImportItemResult(entry="b", status="error", error="x", message="m"),
                ],
            )
            with patch("openjiuwen_plugin.handlers.skill_import", return_value=resp):
                code = main(
                    [
                        "skill-import",
                        str(zip_path),
                        "--market-url",
                        "http://localhost:8000",
                        "--system-token",
                        "t",
                    ]
                )
        self.assertEqual(code, 1)

    def test_skill_import_rejects_oversize_zip_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "big.zip"
            zip_path.write_bytes(b"x" * 20)
            with patch("openjiuwen_plugin.handlers.SKILL_IMPORT_BUNDLE_MAX_BYTES", 10):
                code = main(
                    [
                        "skill-import",
                        str(zip_path),
                        "--market-url",
                        "http://localhost:8000",
                        "--system-token",
                        "t",
                    ]
                )
        self.assertEqual(code, 1)

    def test_skill_import_empty_directory_pack_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty-bundle"
            empty.mkdir()
            with patch("openjiuwen_plugin.handlers.skill_import") as m:
                code = main(
                    [
                        "skill-import",
                        str(empty),
                        "--market-url",
                        "http://localhost:8000",
                        "--system-token",
                        "t",
                    ]
                )
                m.assert_not_called()
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
