import json
import os
import random
import time
import logging
from typing import Dict, List, Optional, OrderedDict, Tuple, Union
import requests
import socket
import requests.packages.urllib3.util.connection as connection

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ContentItem(BaseModel):
    text: Optional[str] = Field(default=None, description='文本内容')
    image: Optional[str] = Field(default=None, description='图片URL')


class Message(BaseModel):
    role: str = Field(..., description='消息角色（如user、assistant）')
    content: List[ContentItem] = Field(..., description='消息内容列表，包含文本或图片')


class BaseTool:
    def _verify_json_format_args(self, params: Union[str, dict]) -> dict:
        """
        验证并格式化参数：支持字符串/json字符串/字典格式输入，转为标准字典
        """
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError as e:
                logger.error(f"参数格式错误，无法解析为JSON: {e}")
                raise ValueError(f"无效的参数格式，需为JSON字符串或字典: {e}")
        if not isinstance(params, dict):
            logger.error(f"参数类型错误，预期str/dict，实际得到: {type(params)}")
            raise TypeError(f"无效的参数类型，需为JSON字符串或字典")
        return params


def register_tool(tool_name: str, allow_overwrite: bool = False):
    def decorator(cls):
        return cls
    return decorator


def extract_images_from_messages(messages: List[Message]) -> List[str]:
    """
    从消息列表中提取所有图片URL，与原qwen_agent函数功能一致
    """
    image_urls = []
    if not messages:
        return image_urls
    for msg in messages:
        for content_item in msg.content:
            if content_item.image:  # 提取非空的图片URL
                image_urls.append(content_item.image)
    return image_urls


DEBUG_MODE = False  # 关闭调试（仅显示最终结果，不显示任何【调试】信息）

def debug_print(content: str):
    """仅用于调试信息的打印，受DEBUG_MODE开关控制，不影响最终结果输出"""
    if DEBUG_MODE:
        print(content)


SERPAPI_IMAGE_SEARCH_KEY = "8888888"
QWEN_IMAGE_SEARCH_MAX_RETRY_TIMES = int(os.getenv('QWEN_IMAGE_SEARCH_MAX_RETRY_TIMES', '3'))
SERPAPI_URL = 'https://serpapi.com/search.json'

# 强制使用IPv4
_orig_getaddrinfo = socket.getaddrinfo
def _new_getaddrinfo(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = _new_getaddrinfo


class ImageResult(BaseModel):
    """Represents an image search result with URL, title, and metadata."""
    id: str = Field(..., description='Unique identifier for the image')
    title: str = Field(..., description='Title or caption of the image')
    imgurl: str = Field(..., description='Direct URL to the image')
    url: str = Field(..., description='Source page URL where the image was found')
    width: str = Field(..., description='Image width in pixels')
    height: str = Field(..., description='Image height in pixels')
    content: str = Field(default='', description='Additional content or description')

    def __str__(self):
        result = {}
        if self.id:
            result['id'] = self.id
        if self.title:
            result['title'] = self.title
        if self.imgurl:
            result['imgurl'] = self.imgurl
        if self.content:
            result['description'] = self.content
        return json.dumps(result, ensure_ascii=False)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def serper_reverse_image_search(image_url: str, check_accessibility: bool = False, max_retry: int = QWEN_IMAGE_SEARCH_MAX_RETRY_TIMES) -> List[ImageResult]:
    """
    输入图片URL，进行反向相似图片搜索（和最初功能一致，使用google_reverse_image引擎）
    Args:
        image_url: 待搜索的图片URL（公开、可访问）
        check_accessibility: 是否检查图片URL可达性
        max_retry: 最大重试次数
    Returns:
        相似图片结果列表
    """
    debug_print(f"\n【调试】serper_reverse_image_search：进入函数，开始图片URL反向搜索")
    debug_print(f"【调试】serper_reverse_image_search：传入参数 -> image_url={image_url}，check_accessibility={check_accessibility}，max_retry={max_retry}")

    # 验证环境变量
    if not SERPAPI_IMAGE_SEARCH_KEY:
        debug_print(f"【调试】serper_reverse_image_search：错误！SERPAPI_IMAGE_SEARCH_KEY 为空，未获取到环境变量")
        raise ValueError(
            'SERPAPI_IMAGE_SEARCH_KEY is None! Please Apply for an apikey from https://serper.dev and set it as an environment variable.'
        )


    payload = {
        'engine': 'google_reverse_image',
        'image_url': image_url,
        'api_key': SERPAPI_IMAGE_SEARCH_KEY,
        'hl': 'zh-CN',
        'gl': 'cn',
        'no_cache': 'true'
    }
    debug_print(f"【调试】serper_reverse_image_search：构造的请求payload -> {json.dumps(payload, ensure_ascii=False)}")

    for retry_count in range(max_retry):
        success = False
        start_time = time.perf_counter()
        response = None
        try:
            debug_print(f"【调试】serper_reverse_image_search：第 {retry_count+1}/{max_retry} 次请求，正在访问 {SERPAPI_URL}")
            # 发送请求
            response = requests.get(SERPAPI_URL, params=payload)
            debug_print(f"【调试】serper_reverse_image_search：请求响应状态码 -> {response.status_code}")

            # 验证响应状态
            response.raise_for_status()

            # 解析JSON响应
            json_response = response.json()
            debug_print(f"【调试】serper_reverse_image_search：解析到JSON响应 -> {json.dumps(json_response, ensure_ascii=False, indent=2)[:800]}...")

            # 提取相似图片结果
            items_data = json_response.get('image_results', []) + json_response.get('inline_images', [])
            debug_print(f"【调试】serper_reverse_image_search：从响应中提取到 image_results + inline_images 共 {len(items_data)} 条数据")

            results: Dict[str, ImageResult] = OrderedDict()
            for idx, item_data in enumerate(items_data):
                try:
                    image_direct_url = item_data.get('original', item_data.get('thumbnail'))
                    source_page_url = item_data.get('link', '')
                    debug_print(f"【调试】serper_reverse_image_search：正在解析第 {idx+1} 条相似图片数据，原始URL={image_direct_url}")

                    if not image_direct_url:
                        debug_print(f"【调试】serper_reverse_image_search：第 {idx+1} 条图片数据无有效URL，跳过")
                        continue

                    # 构造ImageResult
                    image = ImageResult(
                        id=str(item_data.get('position', idx)),
                        title=item_data.get('title', ''),
                        imgurl=image_direct_url,
                        url=source_page_url,
                        width=item_data.get('width', ''),
                        height=item_data.get('height', ''),
                        content=item_data.get('snippet', '') or item_data.get('description', '')
                    )

                    # 去重逻辑
                    if image.imgurl in results and len(results[image.imgurl].title) > len(image.title):
                        debug_print(f"【调试】serper_reverse_image_search：发现重复图片 {image.imgurl}，跳过")
                        continue
                    else:
                        if check_accessibility:
                            _, is_accessible = check_image_url_accessibility(image.imgurl)
                            debug_print(f"【调试】serper_reverse_image_search：图片 {image.imgurl} 可达性检查结果 -> {is_accessible}")
                            if is_accessible:
                                results[image.imgurl] = image
                                debug_print(f"【调试】serper_reverse_image_search：图片 {image.imgurl} 已加入结果集")
                            else:
                                debug_print(f"【调试】serper_reverse_image_search：图片 {image.imgurl} 不可达，跳过")
                        else:
                            results[image.imgurl] = image
                            debug_print(f"【调试】serper_reverse_image_search：图片 {image.imgurl} 直接加入结果集（未做可达性检查）")
                    success = True
                except Exception as e:
                    logger.warning(f"Failed to parse image item: {e}")
                    debug_print(f"【调试】serper_reverse_image_search：解析第 {idx+1} 条图片数据失败，错误 -> {e}")
                    continue


            final_results = [x for x in results.values()]
            debug_print(f"【调试】serper_reverse_image_search：最终整理出 {len(final_results)} 条有效相似图片结果")
            return final_results

        except Exception as e:
            response_text = response.text if response and response.text else "无响应内容"
            logger.error(f'image_search_fail, Error: {e}')
            debug_print(f"【调试】serper_reverse_image_search：第 {retry_count+1} 次请求失败，错误 -> {e}，响应内容 -> {response_text[:500]}...")
            time.sleep(random.uniform(0.1, 1))
        finally:
            cost_time = int((time.perf_counter() - start_time) * 1000)
            debug_print(f"【调试】serper_reverse_image_search：第 {retry_count+1} 次请求耗时 {cost_time} 毫秒")

    debug_print(f"【调试】serper_reverse_image_search：所有重试次数耗尽，返回空列表")
    return []


@register_tool('reverse_image_search', allow_overwrite=True)
class ReverseImageSearch(BaseTool):
    name = 'reverse_image_search'
    description = 'Reverse image search by image URL, input image URL to find similar images with information.'
    parameters = {
        'type': 'object',
        'properties': {
            'img_idx': {
                'type': 'number',
                'description': 'The index of the image (starting from 0) in the messages'
            }
        },
        'required': ['img_idx']
    }

    def call(self, params: Union[str, dict], **kwargs) -> List[ContentItem]:
        debug_print(f"\n【调试】ReverseImageSearch.call：进入工具调用方法")
        # 验证并格式化参数
        params = self._verify_json_format_args(params)
        debug_print(f"【调试】ReverseImageSearch.call：格式化后的参数 -> {params}")

        # 获取图片索引
        image_id = int(params.get('img_idx', 0))
        # 从消息中提取所有图片URL
        images = extract_images_from_messages(kwargs.get('messages', []))
        debug_print(f"【调试】ReverseImageSearch.call：从消息中提取到 {len(images)} 张图片，指定图片索引 {image_id}")

        # 校验图片列表
        if not images:
            error_msg = 'Error: no images found in the messages (no image URL input).'
            debug_print(f"【调试】ReverseImageSearch.call：{error_msg}")
            return [ContentItem(text=error_msg)]
        if image_id >= len(images):
            image_id = len(images) - 1
            debug_print(f"【调试】ReverseImageSearch.call：指定索引超出范围，自动调整为 {image_id}")

        # 获取待搜索的图片URL
        target_image_url = images[image_id]
        debug_print(f"【调试】ReverseImageSearch.call：最终选定的待搜索图片URL -> {target_image_url}")

        try:
            # 调用图片反向搜索函数
            search_results = serper_reverse_image_search(image_url=target_image_url, check_accessibility=False)
            debug_print(f"【调试】ReverseImageSearch.call：serper_reverse_image_search 返回 {len(search_results)} 条相似图片结果")

            content = []
            for i, r in enumerate(search_results):
                txt = f'[{str(i+1)}] "{r.imgurl}" {r.title}\n{r.content}'
                txt = txt.strip('\n')
                debug_print(f"【调试】ReverseImageSearch.call：正在整理第 {i+1} 条结果，文本内容 -> {txt[:100]}...")

                if txt:
                    content.append(ContentItem(text=txt))
                if r.imgurl:
                    content.append(ContentItem(image=r.imgurl))

            debug_print(f"【调试】ReverseImageSearch.call：最终整理出 {len(content)} 条返回内容")
            return content
        except Exception as e:
            error_msg = f'Exception in ReverseImageSearch.call: {repr(e)}'
            logger.info(error_msg)
            debug_print(f"【调试】ReverseImageSearch.call：捕获异常 -> {error_msg}")
            return [ContentItem(text=f'Error: {repr(e)}')]


def check_image_url_accessibility(url: str, timeout: int = 10) -> Tuple[str, bool]:
    """Check if an image URL is accessible (synchronous version)"""
    debug_print(f"【调试】check_image_url_accessibility：正在检查URL可达性 -> {url}，超时时间 {timeout} 秒")
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        debug_print(f"【调试】check_image_url_accessibility：URL {url} 响应状态码 -> {response.status_code}")
        return url, response.status_code == 200
    except Exception as e:
        logger.debug(f"Image URL not accessible: {url}, error: {e}")
        debug_print(f"【调试】check_image_url_accessibility：URL {url} 不可达，错误 -> {e}")
        return url, False

def message_search(
    img_url: str,
    search_text: str = "",
    hl: str = "zh-CN",
    gl: str = "cn",
    no_cache: bool = True,
    num: int = 10
):
    """
    封装ReverseImageSearch的call方法，仅支持字符串/数字类型输入，简化调用流程
    :param img_url: 待搜索的图片URL（字符串）
    :param search_text: 辅助搜索的文字内容（字符串）
    :param hl: 搜索语言配置（字符串，默认"zh-CN"）
    :param gl: 搜索地区配置（字符串，默认"cn"）
    :param no_cache: 是否禁用缓存
    :param num: 返回结果数量（可选，数字/int，默认10）
    :return: 与ReverseImageSearch().call()一致的返回结果
    """
    # 步骤1：参数合法性校验
    if not img_url or not isinstance(img_url, str):
    #if not isinstance(img_url, str):
        raise ValueError("【错误】img_url不能为空，且必须为字符串类型的有效URL")
    if not isinstance(search_text, str):
        raise ValueError("【错误】search_text必须为字符串类型")
    if not isinstance(hl, str) or not isinstance(gl, str):
        raise ValueError("【错误】hl（语言）、gl（地区）必须为字符串类型")
    if not isinstance(no_cache, bool):
        raise ValueError("【错误】no_cache必须为bool型，且仅支持\"true\"或\"false\"")
    if not isinstance(num, int) or num <= 0:
        raise ValueError("【错误】num必须为大于0的数字（int类型）")
    
    # 步骤2：构造payload
    if img_url == "":
        payload = {
            'engine': 'google_reverse_image',  # 固定值，无需用户配置
            'api_key': SERPAPI_IMAGE_SEARCH_KEY,  # 常量，无需用户配置
            'hl': hl,  # 拆分后的字符串参数
            'gl': gl,  # 拆分后的字符串参数
            'no_cache': str(no_cache),  # 拆分后的字符串参数
            'num': num  # 拆分后的数字参数
        }
    else:
        payload = {
            'engine': 'google_reverse_image',  # 固定值，无需用户配置
            'image_url': img_url,  # 传入的图片URL（字符串）
            'api_key': SERPAPI_IMAGE_SEARCH_KEY,  # 常量，无需用户配置
            'hl': hl,  # 拆分后的字符串参数
            'gl': gl,  # 拆分后的字符串参数
            'no_cache': str(no_cache),  # 拆分后的字符串参数
            'num': num  # 拆分后的数字参数
        }
    
    # 步骤3：打印调试payload
    debug_print(f"【调试】构造的请求payload -> {json.dumps(payload, ensure_ascii=False)}")
    
    # 步骤4：构造call方法所需的messages参数
    content_items = [ContentItem(image=img_url)]  # 图片内容
    if search_text:  # 文字内容
        content_items.append(ContentItem(text=search_text))
    
    messages = [
        Message(
            role='user',
            content=content_items
        )
    ]
    
    # 步骤5：构造params参数
    params = {'img_idx': 0}
    
    # 步骤6：调用原ReverseImageSearch的call方法并返回结果
    try:
        debug_print("【调试】开始执行图片反向搜索...")
        search_result = ReverseImageSearch().call(
            params=params,
            messages=messages
        )
        debug_print("【调试】图片反向搜索执行完成")
        return search_result
    except Exception as e:
        debug_print(f"【错误】图片反向搜索执行失败：{str(e)}")
        raise

if __name__ == '__main__':
     # 调试信息
    debug_print("【调试】程序开始运行，初始化 ReverseImageSearch 工具（图片URL反向搜索）")

    # 1. 基础使用
    test_image_url = "https://img10.360buyimg.com/n4/s330x330_jfs/t1/359786/35/3306/92151/6909cd6eFe97c7548/23b08acbaadc3464.jpg"
    search_text="电子产品 商品图片 相似款"
    #test_image_url = ""
    #search_text="火影忍者"

    # 2. 完整使用
    complete_result = message_search(
        img_url=test_image_url,  
        search_text=search_text,  
        hl="zh-CN",  
        gl="cn", 
        no_cache=False,  
        num=20  
    )

   
    print("\n" + "=" * 80)
    print("【最终输出】图片URL反向搜索（相似图片）结果：")
    print("-" * 80)


    test_result = complete_result
    if not test_result:
        print("【提示】未获取到相似图片结果（可更换公开知名图片URL重试）")
    else:
        for item in test_result:
            if item.text:
                print(f"文本内容：{item.text}")
            if item.image:
                print(f"相似图片URL：{item.image}")
            print("-" * 80)
    print("=" * 80)
"""
payload = {
        'engine': 'google_reverse_image',
        'image_url': image_url,
        'api_key': SERPAPI_IMAGE_SEARCH_KEY,
        'hl': 'zh-CN',
        'gl': 'cn',
        'no_cache': 'true'
    }
    debug_print(f"【调试】serper_reverse_image_search：构造的请求payload -> {json.dumps(payload, ensure_ascii=False)}")
"""