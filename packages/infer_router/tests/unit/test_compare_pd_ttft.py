"""Tests for the closed TTFT comparison script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "tools" / "compare_pd_ttft.py"
SPEC = importlib.util.spec_from_file_location("compare_pd_ttft", SCRIPT)
assert SPEC and SPEC.loader
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def test_ttft_breakdown_closes_and_exposes_hidden_saving(tmp_path):
    hixl_log = tmp_path / "hixl.log"
    hixl_log.write_text(
        "\n".join(
            (
                "2026-09-16 10:00:00 [KVBMPDTiming] request_id=h1 "
                "route_ms=700 p_request_ms=500 pd_import_ms=200 "
                "router_to_early_d_dispatch_ms=800",
                "2026-09-16 10:00:01 [Complete] h1 workers=['d1'] "
                "ttft=1700ms total=3000ms tokens=100/10/110",
                "2026-09-16 10:00:01 [KVBMRouterTiming] request=h1 "
                "stage=complete target=d1 phase_ms={'router_route_ms': 650, "
                "'router_capacity_wait_ms': 300, "
                "'worker_dispatch_to_first_byte_ms': 900, "
                "'client_ttft_ms': 1700}",
            )
        ),
        encoding="utf-8",
    )
    mooncake_log = tmp_path / "mooncake.log"
    mooncake_log.write_text(
        "\n".join(
            (
                "2026-09-16 11:00:00 [MooncakePDTiming] request_id=m1 "
                "route_ms=200 p_request_ms=800",
                "2026-09-16 11:00:01 [MooncakePDDispatchTiming] request_id=m1 "
                "d_selection_to_dispatch_ms=800 "
                "p_complete_to_d_dispatch_ms=5 snapshot_callback_ms=10",
                "2026-09-16 11:00:03 [Complete] m1 workers=['d1'] "
                "ttft=3300ms total=5000ms tokens=100/10/110",
                "2026-09-16 11:00:03 [KVBMRouterTiming] request=m1 "
                "stage=complete target=d1 phase_ms={'router_route_ms': 180, "
                "'worker_dispatch_to_first_byte_ms': 2200, "
                "'client_ttft_ms': 3300}",
            )
        ),
        encoding="utf-8",
    )

    hixl = comparison.build_breakdown(
        "prefix-hixl", comparison.parse_router_log(hixl_log), "hixl"
    )
    mooncake = comparison.build_breakdown(
        "prefix-mc", comparison.parse_router_log(mooncake_log), "mooncake"
    )
    output = comparison.compare_breakdowns(hixl, mooncake)

    assert hixl.mean_component("route") == 700
    assert hixl.mean_component("route_to_dispatch") == 100
    assert hixl.mean_component("dispatch_to_first_byte") == 900
    assert hixl.mean_component("residual") == 0
    assert mooncake.mean_component("route") == 200
    assert mooncake.mean_component("route_to_dispatch") == 805
    assert mooncake.mean_component("dispatch_to_first_byte") == 2200
    assert mooncake.mean_component("residual") == 95
    assert "**+1.600s**" in output
    assert "误差 +0.000s" in output
    assert "路由完成 -> D HTTP派发" in output
    assert "嵌套诊断指标（不可与上表相加）" in output


def test_mooncake_breakdown_falls_back_to_p_request(tmp_path):
    log = tmp_path / "mooncake.log"
    log.write_text(
        "\n".join(
            (
                "[MooncakePDTiming] request_id=m1 route_ms=100 p_request_ms=400",
                "[Complete] m1 workers=['d1'] ttft=1000ms total=1200ms "
                "tokens=100/10/110",
                "[KVBMRouterTiming] request=m1 stage=complete target=d1 "
                "phase_ms={'worker_dispatch_to_first_byte_ms': 500}",
            )
        ),
        encoding="utf-8",
    )

    run = comparison.build_breakdown(
        "prefix-mc", comparison.parse_router_log(log), "mooncake"
    )

    assert run.sources["p_request_plus_dispatch_gap"] == 1
    assert run.mean_component("route_to_dispatch") == 400
    assert run.mean_component("residual") == 0
