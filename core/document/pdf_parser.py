"""
PDF解析器模块
原代码位置：Rebuild/Fss_analyzer.py 中的 extract_pdf_content 函数

功能：
- 从PDF文档中提取图像和文本
- 提供通用的文档解析接口

重构说明：
- 将PDF解析功能独立出来
- 为未来支持其他文档格式（Word、图片等）留出接口
"""

import fitz  # PyMuPDF
import io
from PIL import Image
from typing import List, Dict, Tuple, Any


class PDFParser:
    """
    PDF解析器类
    原函数：Rebuild/Fss_analyzer.py 中的 extract_pdf_content
    
    用于从PDF文档中提取图像和文本内容
    """
    
    def extract_content(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        提取PDF中的所有图像和文本内容
        
        原函数：Rebuild/Fss_analyzer.py 中的 extract_pdf_content
        
        参数:
            pdf_path: PDF文件路径
        
        返回:
            (images, full_text) 元组
            - images: 图像列表，每个元素包含 {'image': PIL.Image, 'page_num': int, 'img_index': int}
            - full_text: 完整文本内容
        """
        try:
            doc = fitz.open(pdf_path)
            images = []
            full_text = ""
            
            # 原代码逻辑：遍历每一页
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 提取文本（原代码逻辑）
                page_text = page.get_text()
                full_text += f"\n=== 第 {page_num + 1} 页 ===\n{page_text}\n"
                
                # 提取图像（原代码逻辑）
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        if image.mode == "RGBA":
                            image = image.convert("RGB")
                        images.append(
                            {
                                "image": image,
                                "page_num": page_num + 1,
                                "img_index": img_index + 1,
                            }
                        )
                    except Exception as e:
                        print(f"图像处理错误 (页{page_num + 1}, 图{img_index + 1}): {e}")
            
            return images, full_text
        except Exception as e:
            print(f"PDF内容提取错误: {e}")
            return [], ""


# 通用文档解析接口（为未来扩展其他格式留出接口）
class DocumentParser:
    """
    通用文档解析器接口
    为未来支持其他文档格式（Word、图片等）提供统一接口
    """
    
    def extract_content(self, file_path: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        提取文档中的图像和文本内容（通用接口）
        
        参数:
            file_path: 文档文件路径
        
        返回:
            (images, full_text) 元组
        """
        # 根据文件扩展名选择对应的解析器
        if file_path.lower().endswith('.pdf'):
            parser = PDFParser()
            return parser.extract_content(file_path)
        # 未来可以扩展其他格式：
        # elif file_path.lower().endswith(('.doc', '.docx')):
        #     parser = WordParser()
        #     return parser.extract_content(file_path)
        # elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        #     parser = ImageParser()
        #     return parser.extract_content(file_path)
        else:
            raise ValueError(f"不支持的文档格式: {file_path}")
