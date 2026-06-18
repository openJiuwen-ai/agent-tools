from vllm.v1.engine.input_processor import InputProcessor
from jiuwen_vllm_affinity.kv_cache_plugin.v1.engine.core import unpack_sharing_cache_salt

_process_inputs = InputProcessor.process_inputs


def process_inputs_jiuwen(self, *args, **kwargs):
    req = _process_inputs(self, *args, **kwargs)
    request_id = args[0] if args else kwargs.get("request_id")
    if request_id is not None and unpack_sharing_cache_salt(request_id) is not None:
        req.cache_salt = None
    return req


def register_input_processor():
    InputProcessor.process_inputs = process_inputs_jiuwen
