# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

"""
天气查询插件路由
"""
from openjiuwen_plugin_server.routers.image_search import message_search
from fastapi import HTTPException, Query

from . import BasePluginRouter

demo_router = BasePluginRouter(
    name="demo",
    description="your_demo_tool_description",
)

@demo_router.router.get("/run")
async def run_demo(
    # 可选参数：设置默认值
    query: str = Query(default="默认关键词", description="查询关键词，用于检索相关信息"),
    num: int = Query(default=10, ge=1, le=100, description="查询数量限制，取值范围1-100")
):
    try:
        # 1. 业务逻辑保持不变（此处是返回query参数）
        business_data = {
            "result": "success",
            "query": query,
            "num": num
        }
        print(f"business_data = {business_data}")
        
        # 2. 按照Agent Studio插件规范，封装标准响应格式
        # 必须包含 code、message、data 核心字段（平台硬性要求）
        return {
            "code": 200,  # 200 表示插件内部业务处理成功
            "message": "插件执行成功",  # 人性化的响应描述
            "data": business_data  # 业务数据放入data字段中，可自定义结构
        }
    except Exception as e:
        # 3. 异常场景也封装符合规范的响应，或抛出符合要求的HTTPException
        error_detail = {
            "code": 422,  # 可自定义业务错误码
            "message": f"插件执行失败：{str(e)}",
            "data": None
        }
        raise HTTPException(
            status_code=422,
            detail=error_detail
        ) from e

# 新增 /search 接口功能
@demo_router.router.get("/search")
async def search_demo(

    img_url: str = Query(default="", description="搜索图片URL（字符串类型）"),
    search_text: str = Query(default="", description="搜索文字（字符串类型，可选，默认空）"),
    hl: str = Query(default="zh-CN", description="语言（字符串类型，默认zh-CN）"),
    gl: str = Query(default="cn", description="地区（字符串类型，默认cn）"),
    no_cache: bool = Query(default=True,description="是否禁用缓存默认true"),
    num: int =Query(default=10,description="返回结果数量（数字类型，int，默认10）")

):
    try:
        
        complete_result = message_search(
            img_url=img_url, 
            search_text=search_text,  
            hl=hl,  
            gl=gl,  
            no_cache=no_cache,  
            num=num  
        )
        print(f"complete_data = {complete_result}")
       
        return {
            "code": 200,  
            "message": "搜索成功",  
            "data": complete_result  
        }
    except Exception as e:
       
        error_detail = {
            "code": 422,
            "message": f"搜索失败：{str(e)}",
            "data": None
        }
        raise HTTPException(
            status_code=422,
            detail=error_detail
        ) from e

# 注册原有端点信息
demo_router.register_endpoint("GET", "/run", run_demo, "run demo")

# 注册新增的 /search 端点信息（必须注册，否则插件平台无法识别）
demo_router.register_endpoint("GET", "/search", search_demo, "search weather information")