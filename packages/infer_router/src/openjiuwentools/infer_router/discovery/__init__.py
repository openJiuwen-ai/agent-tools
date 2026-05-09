"""工作器发现模块"""

from openjiuwentools.infer_router.discovery.base import WorkerDiscovery
from openjiuwentools.infer_router.discovery.config_discovery import ConfigDiscovery
from openjiuwentools.infer_router.discovery.etcd_discovery import EtcdDiscovery

__all__ = ["WorkerDiscovery", "ConfigDiscovery", "EtcdDiscovery"]
