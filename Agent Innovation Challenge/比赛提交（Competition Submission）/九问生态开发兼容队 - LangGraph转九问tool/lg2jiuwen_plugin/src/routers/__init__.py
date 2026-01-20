# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

"""
路由模块 - 按插件功能分层管理API路由
"""
from typing import Dict, Any, List

from fastapi import APIRouter

class BasePluginRouter:
    """插件路由基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.router = APIRouter(tags=[name])
        self.endpoints: List[Dict[str, Any]] = []

    def register_endpoint(self, method: str, path: str, func, description: str, **kwargs):
        """注册端点信息"""
        endpoint_info = {
            "method": method.upper(),
            "path": path,
            "function": func.__name__,
            "description": description,
            **kwargs
        }
        self.endpoints.append(endpoint_info)

    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            "name": self.name,
            "description": self.description,
            "prefix": self.router.prefix,
            "endpoints": self.endpoints,
            "endpoint_count": len(self.endpoints)
        }

# 导出所有路由模块
from .lg2jiuwen_router import lg2jiuwen_router

# 所有路由器列表
ALL_ROUTERS = [
    lg2jiuwen_router,
]