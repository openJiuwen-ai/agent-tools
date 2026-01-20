# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

"""
天气查询插件路由
"""

from fastapi import HTTPException, Query, Body

from src.routers import BasePluginRouter
from src.lg2jiuwen_tool import service as lg2jiuwen_service

lg2jiuwen_router = BasePluginRouter(
    name="lg2jiuwen",
    description="LangGraph Agent to Open Jiuwen Agent",
)

@lg2jiuwen_router.router.post("/migrate")
async def migrate(
    source_path: str = Body(..., description="source code path"),
    output_dir: str = Body(..., description="output dir"),
):
    try:
        return {
            "result": "success",
            "data": lg2jiuwen_service.migrate(source_path, output_dir)
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"migrate failed: {str(e)}"
        ) from e

@lg2jiuwen_router.router.get("/run")
async def run(
    source_path: str = Query(..., description="source code path"),
):
    try:
        return {
            "result": "success",
            "data": lg2jiuwen_service.run(source_path),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"run failed: {str(e)}"
        ) from e


@lg2jiuwen_router.router.get("/get_file_content")
async def get_file_content(
    file_path: str = Query(..., description="file path"),
):
    try:
        return {
            "result": "success",
            "data": lg2jiuwen_service.get_file_content(file_path),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"get file content failed: {str(e)}"
        ) from e

# 注册端点信息
lg2jiuwen_router.register_endpoint("GET", "/run", run, "lg2jiuwen run python file")
lg2jiuwen_router.register_endpoint("POST", "/migrate", migrate, "lg2jiuwen migrate file")
lg2jiuwen_router.register_endpoint("GET", "/get_file_content", get_file_content, "lg2jiuwen get file content")