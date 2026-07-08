from types import SimpleNamespace

from jiuwen_vllm_affinity.kv_cache_plugin.v1.request import (
    request_get_session_binding_id,
    request_get_sharing_cache_salt,
)


def test_session_binding_prefers_sharing_cache_salt():
    req = SimpleNamespace(
        sharing_cache_salt="shared-session",
        cache_salt="isolated-session",
    )
    assert request_get_session_binding_id(req) == "shared-session"


def test_session_binding_falls_back_to_cache_salt():
    req = SimpleNamespace(cache_salt="isolated-session")
    assert request_get_sharing_cache_salt(req) is None
    assert request_get_session_binding_id(req) == "isolated-session"


def test_session_binding_empty_when_no_salt():
    req = SimpleNamespace(cache_salt=None)
    assert request_get_session_binding_id(req) is None

    req_empty = SimpleNamespace(cache_salt="")
    assert request_get_session_binding_id(req_empty) is None
