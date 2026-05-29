"""
结构文档分析器模块（支持FSS、超材料、metasurface等）
原代码位置：Rebuild/Fss_analyzer.py 中的 analyze_fss_pdf 及相关函数

功能：
- 使用大模型分析结构文档（FSS、超材料、metasurface等）
- 提取结构周期参数
- 识别结构图
- 分析颜色和材料映射

重构说明：
- 将原 analyze_fss_pdf 函数重构为 StructureAnalyzer 类
- 使用 LLMClient 进行大模型调用
- 使用 PDFParser 进行文档解析
- 保持原有逻辑和算法不变
- 重命名：从 FSSAnalyzer 改为 StructureAnalyzer，以支持更广泛的结构类型
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image

from core.document.llm_client import LLMClient
from core.document.pdf_parser import PDFParser

# 原代码：标准化的颜色范围定义（HSV颜色空间）
STANDARD_COLOR_RANGES = {
    'black': [((0, 0, 0), (180, 255, 30))],
    'gray': [((0, 0, 46), (180, 43, 220))],
    'white': [((0, 0, 221), (180, 30, 255))],
    'red': [
        ((0, 43, 46), (10, 255, 255)),
        ((156, 43, 46), (180, 255, 255))
    ],
    'orange': [((11, 43, 46), (25, 255, 255))],
    'yellow': [((26, 43, 46), (34, 255, 255))],
    'green': [((35, 43, 46), (77, 255, 255))],
    'cyan': [((78, 43, 46), (99, 255, 255))],
    'blue': [((100, 43, 46), (124, 255, 255))],
    'purple': [((125, 43, 46), (155, 255, 255))]
}


class StructureAnalyzer:
    """
    结构文档分析器类（支持FSS、超材料、metasurface等）
    原函数：Rebuild/Fss_analyzer.py 中的 analyze_fss_pdf
    
    用于分析结构文档（FSS、超材料、metasurface等），提取结构图、周期参数、颜色材料映射等信息
    
    重命名说明：
    - 原类名：FSSAnalyzer
    - 新类名：StructureAnalyzer
    - 原因：代码不仅处理FSS，还支持超材料、metasurface等多种结构类型
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, 
                 pdf_parser: Optional[PDFParser] = None):
        """
        初始化结构分析器
        
        参数:
            llm_client: 大模型客户端（如果为None，自动创建）
            pdf_parser: PDF解析器（如果为None，自动创建）
        """
        self.llm_client = llm_client if llm_client else LLMClient()
        self.pdf_parser = pdf_parser if pdf_parser else PDFParser()
    
    def extract_structure_parameters_from_text(self, full_text: str) -> Dict[str, str]:
        """
        使用大模型从PDF文本中提取结构周期参数（X、Y、Z）
        支持FSS、超材料、metasurface等结构类型
        
        原函数：Rebuild/Fss_analyzer.py 中的 extract_fss_parameters_from_text
        
        参数:
            full_text: PDF提取的完整文本
        
        返回:
            结构周期参数字典 {'X': str, 'Y': str, 'Z': str, 'Unit': str}
        """
        print("\n" + "=" * 50)
        print("正在从PDF文本中提取结构周期参数（FSS/超材料/metasurface）...")
        print("=" * 50)
        
        # 原代码：参数提取提示词（扩展支持多种结构类型）
        parameter_extraction_prompt = f"""请分析提供的PDF文档文本，提取结构单元的周期尺寸参数（支持FSS、超材料、metasurface等）。

任务要求：
1. 提取FSS单元周期的三个关键尺寸：
   - X：单元在X方向的周期长度
   - Y：单元在Y方向的周期长度
   - Z：单元的厚度（介质板厚度）

2. 参数识别规则：
   - X和Y（长宽周期）：
     * 在FSS设计中，L、D通常表示单元周期
     * 对于对称结构（如Jerusalem cross），X和Y方向周期相同
     * 从文本或者表格中找到标注为单元周期的参数（如L、P、period等），提取参数数值中最大的数值作为X和Y
     * 如果文本描述"unit cell size"、"periodicity"、"lattice constant"，即为周期
     * 如果表格存在D、W、L、Dx、Dy等参数，则提取参数数值中的最大值作为X和Y
     * 禁止输出"unknow"，若未推理出X和Y的数值，则直接将所有英文参数提取出来，将单位为mm的参数的数值中的最大值作为X和Y

   - Z（厚度）：
     * 文档会说明"FSS单元印制在介质板上"
     * 介质板厚度即为Z值
     * 关键词：substrate thickness、dielectric thickness、PCB thickness、h等
     * 如果表格中出现d、h等表示厚度的参数，则提取d、h等厚度参数中的值为Z
     * 如果文中没有提到介质板厚度，则默认输出Z为1.6mm

3. 数据来源优先级：
   - 优先从PDF文本中的表格提取
   - 如果没有表格，从文本描述中提取，例如：
     "The unit cell period is L=23mm"
     "The substrate thickness is h=1.6mm"
     "Periodicity: 10mm × 10mm"
   - 确保单位统一（通常为mm）

4. 返回格式（严格Python字典）：
fss_size = {{
    'X': '23',
    'Y': '23',
    'Z': '1.6',
    'Unit': 'mm'
}}

PDF文本内容：
{full_text[:5000]}

注意事项：
- 只提取周期参数X、Y、Z，不要提取其他结构参数
- 如果X和Y相同（对称结构），两者值相同
- 如果找不到某个参数，使用 'unknown'
- Unit字段记录统一单位（如mm、μm等）
- 数值只保留数字部分，不要包含单位"""
        
        # 原代码：调用大模型API
        api_result = self.llm_client.call_api(
            parameter_extraction_prompt,
            image_data=None,
            max_tokens=6000
        )
        
        if not api_result:
            print("API参数提取失败")
            return {}
        
        # 原代码：解析API返回结果
        try:
            fss_size_match = re.search(r"fss_size\s*=\s*(\{.*?})", api_result, re.DOTALL)
            if fss_size_match:
                fss_size = eval(fss_size_match.group(1))
                # 验证必须包含X、Y、Z、Unit字段
                required_fields = ['X', 'Y', 'Z', 'Unit']
                if all(field in fss_size for field in required_fields):
                    return fss_size
                else:
                    print(f"警告：返回字典缺少必需字段，当前字段：{list(fss_size.keys())}")
                    # 补充缺失字段
                    for field in required_fields:
                        if field not in fss_size:
                            fss_size[field] = 'unknown'
                    return fss_size
            else:
                print("未找到fss_size字典")
                print(f"API返回内容：\n{api_result}")
                return {}
        except Exception as e:
            print(f"参数解析错误: {e}")
            print(f"完整API返回：\n{api_result}")
            return {}
    
    def select_fss_structure_image(self, encoded_images: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        使用大模型逐个判断并选择FSS结构图
        
        原函数：Rebuild/Fss_analyzer.py 中的 select_fss_structure_image
        
        参数:
            encoded_images: 编码后的图像列表，每个元素包含 'base64', 'page_num', 'img_index', 'image'
        
        返回:
            选中的图像信息字典，如果未找到则返回None
        """
        # 原代码：图像选择提示词
        select_prompt = """请判断这张图像是否为结构图。

判断标准（必须同时满足）:
✓ 必须包含：箭头、英文长度参数（如 L、W、P、a、b、g、h、r等）
✓ 必须显示：几何结构（如方形环、十字形、圆环等）

✗ 禁止包含：
  - 频率标注（GHz）
  - dB值标注
  - 曲线图（S参数图、频率响应图等）
  - A/m柱状图
  - 任何图表、坐标轴

返回格式（严格JSON）：
{
    "is_fss_structure": true/false,
    "reason": "判断理由（简短说明）"
}"""
        
        # 原代码：遍历所有图像进行判断
        for img_info in encoded_images:
            print(f"\n正在判断图像：页{img_info['page_num']}, 索引{img_info['img_index']}")
            
            api_result = self.llm_client.call_api(select_prompt, img_info["base64"])
            
            if api_result:
                try:
                    # 尝试提取JSON（原代码逻辑）
                    json_match = re.search(r'\{.*}', api_result, re.DOTALL)
                    if json_match:
                        result_json = json.loads(json_match.group())
                        is_fss = result_json.get("is_fss_structure", False)
                        reason = result_json.get("reason", "无理由")
                        
                        print(f"  判断结果: {'✓ 是FSS结构图' if is_fss else '✗ 不是FSS结构图'}")
                        print(f"  理由: {reason}")
                        
                        if is_fss:
                            print(f"\n>>> 已选择FSS结构图：页{img_info['page_num']}, 索引{img_info['img_index']}")
                            return img_info
                except Exception as e:
                    print(f"  解析API结果错误: {e}")
        
        print("\n!!! 警告：未找到符合标准的FSS结构图")
        return None
    
    def analyze_color_material(self, selected_image: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """
        使用大模型分析FSS结构图的颜色和材料
        
        原函数：Rebuild/Fss_analyzer.py 中 analyze_fss_pdf 函数的颜色材料分析部分
        
        参数:
            selected_image: 选中的FSS结构图信息
        
        返回:
            (color_ranges, col_mats, col_comps) 元组
        """
        # 原代码：生成标准颜色范围字符串
        standard_colors_str = json.dumps(STANDARD_COLOR_RANGES, indent=2, ensure_ascii=False)
        
        # 原代码：颜色材料分析提示词
        color_material_analysis_prompt = f"""请对提供的FSS (Frequency Selective Surface) 结构图进行颜色和材料分析。

严格遵守以下规则：

1. 颜色范围必须严格基于以下标准化HSV颜色空间定义：
{standard_colors_str}

2. 材料类型规则：
   - 所有金属材料 → 标记为 'PEC'
   - 所有介质材料 → 标记为 'FR4'

3. 组件描述应基于FSS结构特点：
   - 辐射单元 (radiating element)
   - 接地层 (ground plane)
   - 介质基板 (dielectric substrate)
   - 等效传输线 (equivalent transmission line)
   等

4. 必须返回Python字典格式（可直接使用），键名和结构不可修改：

color_ranges = {{
    'black': [((0, 0, 0), (180, 255, 30))],
    'white': [((0, 0, 221), (180, 30, 255))],
}}

col_mats = {{
    'black': 'PEC',
    'white': 'FR4',
}}

col_comps = {{
    'black': 'radiating element',
    'white': 'dielectric substrate',
}}

注意：
- 颜色名称必须从标准颜色范围中选择
- 若某字段无数据，用 'unknown' 填充（不可删除字段）
- HSV范围必须严格对应标准定义"""
        
        # 原代码：调用大模型API
        api_result = self.llm_client.call_api(
            color_material_analysis_prompt,
            selected_image["base64"],
            max_tokens=6000
        )
        
        if not api_result:
            print("API颜色材料分析失败")
            return {}, {}, {}
        
        # 原代码：解析API返回结果
        try:
            print("\nAPI返回结果预览：")
            print(api_result[:500])
            print("\n解析结果...")
            
            color_ranges_match = re.search(r"color_ranges\s*=\s*(\{.*?})", api_result, re.DOTALL)
            color_ranges = eval(color_ranges_match.group(1)) if color_ranges_match else {}
            
            col_mats_match = re.search(r"col_mats\s*=\s*(\{.*?})", api_result, re.DOTALL)
            col_mats = eval(col_mats_match.group(1)) if col_mats_match else {}
            
            col_comps_match = re.search(r"col_comps\s*=\s*(\{.*?})", api_result, re.DOTALL)
            col_comps = eval(col_comps_match.group(1)) if col_comps_match else {}
            
            return color_ranges, col_mats, col_comps
        except Exception as e:
            print(f"API结果解析错误: {e}")
            print(f"完整API返回：\n{api_result}")
            return {}, {}, {}
    
    def analyze_fss_document(self, document_path: str, output_folder: str = r"fss_out") -> Tuple[Dict[str, str], Dict[str, str], str]:
        """
        分析FSS文档（主入口函数）
        
        原函数：Rebuild/Fss_analyzer.py 中的 analyze_fss_pdf
        
        参数:
            document_path: 文档路径（目前支持PDF，未来可扩展）
            output_folder: 输出文件夹
        
        返回:
            (col_mats, fss_size, pic_name) 元组
            - col_mats: 颜色-材料映射字典
            - fss_size: FSS周期参数字典
            - pic_name: 保存的结构图文件名
        """
        # 1. 提取文档中的所有图像和文本（原代码逻辑）
        all_images, full_text = self.pdf_parser.extract_content(document_path)
        if not all_images:
            print("无法从文档中提取任何图像")
            return {}, {}, ""
        
        print(f"\n成功提取 {len(all_images)} 张图像")
        print(f"提取文本长度: {len(full_text)} 字符")
        
        # 2. 先进行文本分析，提取FSS周期参数（X、Y、Z）（原代码逻辑）
        fss_size = self.extract_fss_parameters_from_text(full_text)
        
        print("\n" + "=" * 50)
        print("FSS周期参数提取完成")
        print("=" * 50)
        print(f"提取结果: {json.dumps(fss_size, indent=2, ensure_ascii=False)}")
        
        # 3. 转换图像为base64（原代码逻辑）
        encoded_images = []
        for img_info in all_images:
            encoded_img = self.llm_client.encode_image_to_base64(img_info["image"])
            encoded_images.append(
                {
                    "base64": encoded_img,
                    "page_num": img_info["page_num"],
                    "img_index": img_info["img_index"],
                    "image": img_info["image"],
                }
            )
        
        # 4. 使用大模型逐个判断并选择FSS结构图（原代码逻辑）
        print("\n" + "=" * 50)
        print("开始逐个判断图像...")
        print("=" * 50)
        
        selected_image = self.select_fss_structure_image(encoded_images)
        
        if not selected_image:
            print("\n!!! 错误：该文档中未找到符合标准的FSS结构图")
            return {}, fss_size, ""
        
        # 5. 保存选中的FSS结构图（原代码逻辑）
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(
            output_folder,
            f"fss_structure_p{selected_image['page_num']}_i{selected_image['img_index']}.png",
        )
        pic_name = f"fss_structure_p{selected_image['page_num']}_i{selected_image['img_index']}.png"
        selected_image["image"].save(output_path)
        print(f"\n已保存FSS结构图: {output_path}")
        
        # 6. 调用大模型进行颜色和材料分析（原代码逻辑）
        print("\n" + "=" * 50)
        print("正在调用API分析FSS结构图颜色和材料...")
        print("=" * 50)
        
        color_ranges, col_mats, col_comps = self.analyze_color_material(selected_image)
        
        # 原代码：返回 col_mats, fss_size, pic_name
        return col_mats, fss_size, pic_name


# 向后兼容的包装函数（保持原有接口）
def analyze_fss_pdf(pdf_path: str, output_folder: str = r"fss_out") -> Tuple[Dict[str, str], Dict[str, str], str]:
    """
    向后兼容函数：分析FSS PDF文档
    
    原函数：Rebuild/Fss_analyzer.py 中的 analyze_fss_pdf
    
    参数:
        pdf_path: PDF文件路径
        output_folder: 输出文件夹
    
    返回:
        (col_mats, fss_size, pic_name) 元组
    """
    analyzer = FSSAnalyzer()
    return analyzer.analyze_fss_document(pdf_path, output_folder)
