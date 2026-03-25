import logging
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml

from cli.main import main
from cli.market import search_plugins
from cli.plugin import info_plugin, init_plugin, install_plugin_from_zip, pack_plugin, validate_plugin


class PluginCommandsTest(unittest.TestCase):
    def test_init_and_validate_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("demo-plugin", Path(tmp))
            self.assertTrue((plugin_root / "schemas" / "tools.json").exists())
            self.assertTrue((plugin_root / "src" / "demo_plugin" / "plugin.py").exists())
            result = validate_plugin(plugin_root)
            self.assertTrue(result.ok, msg=f"errors: {result.errors}")

    def test_init_and_validate_mcp_stdio_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("demo-mcp", Path(tmp), plugin_type="mcp-stdio")
            self.assertTrue((plugin_root / "schemas" / "tools.json").exists())
            self.assertTrue((plugin_root / "src" / "demo_mcp" / "mcp_server.py").exists())
            self.assertTrue((plugin_root / "pyproject.toml").exists())
            result = validate_plugin(plugin_root)
            self.assertTrue(result.ok, msg=f"errors: {result.errors}")

    def test_validate_fails_with_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "broken"
            root.mkdir(parents=True)
            (root / "plugin.yaml").write_text("name: broken-plugin\nversion: 0.1.0\n", encoding="utf-8")
            result = validate_plugin(root)
            self.assertFalse(result.ok)
            self.assertGreater(len(result.errors), 0)

    def test_validate_detects_tool_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("mismatch-plugin", Path(tmp))
            plugin_py = plugin_root / "src" / "mismatch_plugin" / "plugin.py"
            plugin_py.write_text(
                """from openjiuwen.core.foundation.tool import tool

@tool(name="another-tool", description="x", input_params={})
def another_tool() -> dict:
    return {"ok": True}
""",
                encoding="utf-8",
            )
            result = validate_plugin(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("another-tool" in e for e in result.errors))

    def test_validate_fails_with_invalid_compatibility_specifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("compat-plugin", Path(tmp))
            plugin_yaml = plugin_root / "plugin.yaml"
            content = plugin_yaml.read_text(encoding="utf-8")
            content = content.replace(">=3.11, <3.14", "3.11")
            plugin_yaml.write_text(content, encoding="utf-8")

            result = validate_plugin(plugin_root)
            self.assertFalse(result.ok)
            self.assertTrue(any("compatibility.python" in e for e in result.errors))

    def test_validate_compatibility_extra_keys_not_validated(self) -> None:
        """仅校验 compatibility.python；其它键（如 openjiuwen）CLI 不校验格式。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("compat-extra", Path(tmp))
            plugin_yaml = plugin_root / "plugin.yaml"
            data = yaml.safe_load(plugin_yaml.read_text(encoding="utf-8"))
            assert isinstance(data, dict) and isinstance(data.get("compatibility"), dict)
            data["compatibility"]["openjiuwen"] = "latest"
            plugin_yaml.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            result = validate_plugin(plugin_root)
            self.assertTrue(result.ok, msg=f"errors: {result.errors}")

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
            plugin_root = init_plugin("pack-demo", Path(tmp), plugin_type="mcp-stdio")
            out_dir = Path(tmp) / "out"
            zip_path = pack_plugin(plugin_root, out_dir)
            self.assertTrue(zip_path.exists())
            self.assertEqual(zip_path.name, "pack-demo-0.1.0.zip")
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
            self.assertTrue(any("plugin.yaml" in n for n in names))
            self.assertTrue(any("pack-demo-0.1.0" in n for n in names))

    def test_pack_excludes_plugin_out_directory(self) -> None:
        """mcp/rest 整目录打包时不应包含插件根目录下的 out/（避免历史 zip 被打进新包）。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("skip-mcp", Path(tmp), plugin_type="mcp-stdio")
            (plugin_root / "out").mkdir(parents=True, exist_ok=True)
            (plugin_root / "out" / "stale.zip").write_bytes(b"dummy")
            zip_path = pack_plugin(plugin_root, Path(tmp) / "publish-out")
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
            prefix = "skip-mcp-0.1.0"
            self.assertFalse(any(n.replace("\\", "/").startswith(f"{prefix}/out/") for n in names))

    def test_init_restful_api_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("demo-api", Path(tmp), plugin_type="restful-api")
            self.assertTrue((plugin_root / "schemas" / "tools.json").exists())
            self.assertTrue((plugin_root / "src" / "demo_api" / "rest_api.py").exists())
            self.assertTrue((plugin_root / "pyproject.toml").exists())
            result = validate_plugin(plugin_root)
            self.assertTrue(result.ok, msg=f"errors: {result.errors}")

    def test_install_mcp_stdio_calls_pip_install_dot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("demo-mcp", Path(tmp), plugin_type="mcp-stdio")
            out_dir = Path(tmp) / "out"
            zip_path = pack_plugin(plugin_root, out_dir)

            # 避免真实 pip 安装（网络/环境不稳定）；验证分支选择与命令参数
            with patch("cli.plugin.subprocess.run") as m_run:
                m_run.return_value = None
                installed_root = install_plugin_from_zip(zip_path)

            self.assertTrue((installed_root / "plugin.yaml").exists())
            self.assertEqual(m_run.call_count, 1)
            pip_cmd = m_run.call_args[0][0]
            self.assertIn("-m", pip_cmd)
            self.assertIn("pip", pip_cmd)
            self.assertEqual(pip_cmd[-1], ".")

    def test_install_restful_api_calls_pip_install_dot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("demo-api", Path(tmp), plugin_type="restful-api")
            out_dir = Path(tmp) / "out"
            zip_path = pack_plugin(plugin_root, out_dir)

            with patch("cli.plugin.subprocess.run") as m_run:
                m_run.return_value = None
                installed_root = install_plugin_from_zip(zip_path)

            self.assertTrue((installed_root / "plugin.yaml").exists())
            self.assertEqual(m_run.call_count, 1)
            pip_cmd = m_run.call_args[0][0]
            self.assertIn("-m", pip_cmd)
            self.assertIn("pip", pip_cmd)
            self.assertEqual(pip_cmd[-1], ".")

    def test_info_reads_readme_local(self) -> None:
        """本地插件目录读取（plugin.info_plugin），非 CLI 市场接口。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("info-demo", Path(tmp))
            info = info_plugin(plugin_root)
            self.assertEqual(info.get("name"), "info-demo")
            self.assertEqual(info.get("version"), "0.1.0")
            self.assertIsNotNone(info.get("readme"))
            self.assertIn("info-demo", info["readme"])

    def test_info_from_market(self) -> None:
        """plugin info 通过版本详情 API 拉取摘要字段。"""
        with patch("cli.handlers.get_plugin_version_detail") as m:
            m.return_value = {
                "asset_id": "demo-id",
                "zip_url": "http://example.com/demo.zip",
                "icon_uri": "http://example.com/icon.png",
                "readme_url": "http://example.com/readme.md",
                "changelog_url": "http://example.com/changelog.log",
            }
            code = main(["info", "demo-id", "--version", "1.0.0", "--market-url", "http://localhost:8000"])
            self.assertEqual(code, 0)
            m.assert_called_once()
            call_kw = m.call_args
            self.assertEqual(call_kw[0][1], "demo-id")
            self.assertEqual(call_kw[0][2], "1.0.0")

    def test_publish_with_system_token(self) -> None:
        """publish 使用 --system-token 时应走 X-System-Token 路径。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("publish-sys-demo", Path(tmp))
            out_zip = Path(tmp) / "out"
            zip_path = pack_plugin(plugin_root, out_zip)

            # cli/plugin.py 在导入时把 upload_plugin 固化为 _market_upload_plugin，
            # 因此测试需要 patch 这个别名，才能真正避免网络请求。
            with patch("cli.plugin._market_upload_plugin") as m:
                # 避免交互输入：提供 --system-token 即可
                code = main(
                    [
                        "publish",
                        "--file",
                        str(zip_path),
                        "--user-id",
                        "user-sys",
                        "--system-token",
                        "sys-token-123",
                        "--market-url",
                        "http://localhost:8000",
                    ]
                )
                self.assertEqual(code, 0)

                self.assertEqual(m.call_count, 1)
                # upload_plugin signature:
                # (market_url, user_token, system_token, user_id, zip_path, checksum_sha256=...)
                call_args = m.call_args
                self.assertEqual(call_args[0][0], "http://localhost:8000")
                self.assertIsNone(call_args[0][1])
                self.assertEqual(call_args[0][2], "sys-token-123")
                self.assertEqual(call_args[0][3], "user-sys")

    def test_publish_user_id_from_env(self) -> None:
        """publish 可从 OPENJIUWEN_USER_ID 取得 user_id（省略 --user-id）。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("publish-env-uid", Path(tmp))
            out_zip = Path(tmp) / "out"
            zip_path = pack_plugin(plugin_root, out_zip)

            with patch("cli.plugin._market_upload_plugin") as m:
                with patch.dict(os.environ, {"OPENJIUWEN_USER_ID": "from-env-user"}, clear=False):
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
                call_args = m.call_args
                self.assertEqual(call_args[0][3], "from-env-user")

    def test_delete_rejects_both_token_and_system_token(self) -> None:
        """delete 同时传 --token 与 --system-token 时应直接失败。"""
        with patch("cli.market.delete_plugin") as m:
            code = main(
                [
                    "delete",
                    "demo-id",
                    "--market-url",
                    "http://localhost:8000",
                    "--user-id",
                    "user-001",
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
        """tools 类型：先 build wheel，zip 内仅含 plugin.yaml、tools.json、README.md、dist/*.whl。"""
        with tempfile.TemporaryDirectory() as tmp:
            plugin_root = init_plugin("wheel-demo", Path(tmp))
            out_dir = Path(tmp) / "out"
            zip_path = pack_plugin(plugin_root, out_dir)
            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
            self.assertTrue(any("plugin.yaml" in n for n in names))
            self.assertTrue(any("schemas/tools.json" in n for n in names))
            self.assertTrue(any("README.md" in n for n in names))
            self.assertTrue(any("dist/" in n and n.endswith(".whl") for n in names))

    def test_pack_fails_when_plugin_yaml_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty"
            root.mkdir(parents=True)
            with self.assertRaises(ValueError) as ctx:
                pack_plugin(root)
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
                pack_plugin(root, Path(tmp) / "out")
            self.assertIn("validation failed", str(ctx.exception))

    def test_search_plugins_maps_plugin_list_query_params(self) -> None:
        """search_plugins 透传 PluginListQuery 相关 query 参数。"""
        with patch("cli.market.requests.get") as m:
            m.return_value.status_code = 200
            m.return_value.headers = {"content-type": "application/json"}
            m.return_value.json.return_value = {"data": {"items": [], "total": 0}}
            log = logging.getLogger("test_search")
            search_plugins(
                "http://127.0.0.1:9",
                "keyword",
                log,
                plugin_type="tools",
                publisher_name="Alice",
                asset_id="asset-1",
                asset_type="plugin",
                publisher_id="pub-1",
                page=3,
                page_size=15,
                order_by="create_time",
                asc=False,
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

    def test_search_plugins_asc_sets_desc_false(self) -> None:
        with patch("cli.market.requests.get") as m:
            m.return_value.status_code = 200
            m.return_value.headers = {"content-type": "application/json"}
            m.return_value.json.return_value = {"data": {"items": [], "total": 0}}
            log = logging.getLogger("test_search_asc")
            search_plugins("http://127.0.0.1:9", "", log, order_by="install_count", asc=True)
            self.assertFalse(m.call_args[1]["params"]["desc"])


if __name__ == "__main__":
    unittest.main()
