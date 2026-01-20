#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

"""
Plugin Server - 主应用文件
模块化插件路由架构
"""
import datetime
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 导入模块化路由
from src.routers.lg2jiuwen_router import lg2jiuwen_router


# 创建FastAPI应用
app = FastAPI(
    title="Plugin Server",
    description="plugin server",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器
    """
    error_id = f"error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

    # 记录错误信息（实际项目中应该使用日志系统）
    error_info = {
        "error_id": error_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "path": str(request.url),
        "method": request.method,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc()
    }

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_id": error_id,
            "error": "Internal Server Error",
            "message": "internal server error",
            "timestamp": datetime.datetime.now().isoformat()
        }
    )

# 根路径
@app.get("/")
async def root():
    """API根路径，返回服务基本信息"""
    return {
        "service": "Plugin Server",
        "version": "1.0.0",
        "description": "plugin server",
        "protocols": ["RESTful API"],
        "plugins": [
            {
                "name": "lg2jiuwen",
                "description": "langgraph 转 jiuwen工具",
                "base_path": "/lg2jiuwen"
            },
        ],
        "total_plugins": 2,
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "operational"
    }

# plugin router
app.include_router(lg2jiuwen_router.router, prefix="/lg2jiuwen")