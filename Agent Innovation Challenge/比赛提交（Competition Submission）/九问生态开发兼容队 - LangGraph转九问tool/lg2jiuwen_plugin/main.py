#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import uvicorn
from dotenv import load_dotenv
import os

from src.restful_tool_router import app

# Load environment variables from .env file
load_dotenv(".env")

# 定义 main 函数（供脚本入口调用）
def main():
    # 尝试多种启动方式
    port = int(os.getenv("PORT",8185))

    try:
        # 方法1: 标准方式
        uvicorn.run(app, host="0.0.0.0", port=port)
    except TypeError as e:
        if "loop_factory" in str(e):
            # 方法2: 兼容方式
            import asyncio
            config = uvicorn.Config(app, host="0.0.0.0", port=port)
            server = uvicorn.Server(config)
            asyncio.run(server.serve())
        else:
            raise

if __name__ == "__main__":
    main()