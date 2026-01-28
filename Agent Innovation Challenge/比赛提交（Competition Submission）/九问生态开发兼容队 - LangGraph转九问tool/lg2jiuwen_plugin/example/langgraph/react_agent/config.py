"""
配置：LLM 和 API Keys
"""

from langchain_openai import ChatOpenAI


# LLM 配置
# `LLM_MODEL` is a variable that stores the model name or identifier used for the language model (LLM)
# configuration. In this case, it is set to "glm-4-flash". This model will be used by the ChatOpenAI
# object for natural language processing tasks.
LLM_MODEL_NAME = "glm-4-flash"
LLM_API_KEY = "a2143076169049208e54a83c9900d084.xyq9uiIZxK9AwWx3"
LLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"

# 天气 API
SENIVERSE_API_KEY = "SBM-NCypTmfxznW6X"

# 初始化 LLM
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_key=LLM_API_KEY,
    openai_api_base=LLM_API_BASE,
    temperature=0
)
