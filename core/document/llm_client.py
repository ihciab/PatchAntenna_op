"""
大模型客户端模块
原代码位置：Rebuild/Fss_analyzer.py 中的 call_aliyun_api 函数

功能：
- 封装大模型API调用（阿里云视觉语言模型）
- 支持文本和图像输入
- 提供统一的API接口

重构说明：
- 将原 call_aliyun_api 函数重构为 LLMClient 类
- 保持原有API调用逻辑不变
- 支持未来扩展其他大模型接口
"""

import requests
import json
import io
import base64
from PIL import Image
from typing import Optional, Dict, Any


class LLMClient:
    """
    大模型客户端类
    原函数：Rebuild/Fss_analyzer.py 中的 call_aliyun_api
    
    用于调用大模型API进行文本和图像分析
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        初始化大模型客户端
        
        参数:
            api_url: API地址（如果为None，从config.json读取）
            api_key: API密钥（如果为None，从config.json读取）
        
        原代码逻辑：从config.json读取API配置
        """
        # 原代码：从config.json读取配置
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            print("错误：JSON文件不存在")
            config_data = {}
        except json.JSONDecodeError:
            print("错误：JSON格式无效（可能有语法错误）")
            config_data = {}
        except Exception as e:
            print(f"读取失败：{e}")
            config_data = {}
        
        # 原代码：从配置中读取API信息
        if api_url is None:
            try:
                api_url = config_data.get("agent_api", {}).get("API_URL", "")
                if not bool(api_url):
                    api_url = input("请输入API_URL")
                    config_data.setdefault("agent_api", {})["API_URL"] = api_url
            except Exception:
                api_url = input("请输入API_URL") if api_url is None else api_url
        
        if api_key is None:
            try:
                api_key = config_data.get("agent_api", {}).get("API_KEY", "")
                if not bool(api_key):
                    api_key = input("请输入API_KEY")
                    config_data.setdefault("agent_api", {})["API_KEY"] = api_key
            except Exception:
                api_key = input("请输入API_KEY") if api_key is None else api_key
        
        # 保存配置（原代码逻辑）
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        
        if model_name is None:
            model_name = config_data.get("agent_api", {}).get("MODEL_NAME", "openai/gpt-5")

        self.api_url = normalize_chat_completions_url(api_url)
        self.api_key = api_key
        self.model_name = model_name
        
        if api_url and api_key:
            print(f"当前API_URL为：{api_url}")
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """
        将PIL图像转换为base64编码字符串
        
        原函数：Rebuild/Fss_analyzer.py 中的 encode_image_to_base64
        
        参数:
            image: PIL图像对象
        
        返回:
            base64编码的字符串
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def call_api(self, prompt: str, image_data: Optional[str] = None, 
                 max_tokens: int = 4000, temperature: float = 0.1) -> str:
        """
        调用大模型API进行分析，支持文本和图像输入
        
        原函数：Rebuild/Fss_analyzer.py 中的 call_aliyun_api
        
        参数:
            prompt: 文本提示词
            image_data: base64编码的图像数据（可选）
            max_tokens: 最大token数
            temperature: 温度参数
        
        返回:
            API返回的文本内容
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com",
            "X-OpenRouter-Title": "FSS-Complete-Analysis-Tool",
        }
        
        messages = [{"role": "user", "content": []}]
        
        # 添加文本内容（原代码逻辑）
        text_content = {"type": "text", "text": prompt}
        messages[0]["content"].append(text_content)
        
        # 添加图像内容（如果提供）（原代码逻辑）
        if image_data:
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            }
            messages[0]["content"].append(image_content)
        
        # 原代码：使用阿里云视觉语言模型
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=180)
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
            print(f"API请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return ""
        except Exception as e:
            print(f"API调用错误: {e}")
            return ""


def normalize_chat_completions_url(api_url: Optional[str]) -> str:
    clean = str(api_url or "").strip().rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return clean + "/chat/completions"

