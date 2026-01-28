"""
配置文件
"""

import os
from openjiuwen.core.utils.llm.model_library.openai import OpenAIChatModel

# SSL 配置
os.environ['LLM_SSL_VERIFY'] = 'false'

# 全局变量
MAX_LOOPS = 3

# LLM 配置
LLM_MODEL_NAME = 'glm-4-flash'
LLM_API_KEY = 'a2143076169049208e54a83c9900d084.xyq9uiIZxK9AwWx3'
LLM_API_BASE = 'https://open.bigmodel.cn/api/paas/v4/'
SENIVERSE_API_KEY = 'SBM-NCypTmfxznW6X'

def get_llm():
    """获取 LLM 实例"""
    return OpenAIChatModel(api_key=LLM_API_KEY, api_base=LLM_API_BASE)