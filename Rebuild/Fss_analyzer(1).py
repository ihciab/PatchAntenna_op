import requests
import fitz
import io
import os
import json
import re
from PIL import Image
import base64

try:
    with open("config.json", "r", encoding="utf-8") as f:
        config_data = json.load(f)
except FileNotFoundError:
    print("错误：JSON文件不存在")
except json.JSONDecodeError:
    print("错误：JSON格式无效（可能有语法错误）")
except Exception as e:
    print(f"读取失败：{e}")
else:
    # 无错误时执行
    print("API配置文件读取成功")
    try:
        API_URL = config_data["agent_api"]["API_URL"]
        API_KEY = config_data["agent_api"]["API_KEY"]
        if not bool(API_URL):
            API_URL = input("请输入API_URL")
        if not bool(API_KEY):
            API_KEY = input("请输入API_KEY")
        print(f"当前API_URL为：{API_URL}")
        config_data["agent_api"]["API_URL"] = API_URL
        config_data["agent_api"]["API_KEY"] = API_KEY
    except json.JSONDecodeError:
        print("错误：JSON格式无效（可能有语法错误）")
    else:
        with open("config.json", "w", encoding="utf-8") as f:
            # indent=2：保持JSON格式缩进（美观）；ensure_ascii=False：正确处理中文
            json.dump(config_data, f, indent=2, ensure_ascii=False)

# 标准化的颜色范围定义（HSV颜色空间）
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


def encode_image_to_base64(image):
    """将PIL图像转换为base64编码字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")



def call_openrouter_api(prompt, image_data=None, max_tokens=40000, temperature=0.1):
    """调用OpenRouter API进行分析，支持文本和图像输入"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com",
        "X-Title": "FSS-Complete-Analysis-Tool",
    }

    messages = [{"role": "user", "content": []}]

    # 添加文本内容
    text_content = {"type": "text", "text": prompt}
    messages[0]["content"].append(text_content)

    # 添加图像内容（如果提供）
    if image_data:
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"},
        }
        messages[0]["content"].append(image_content)

    data = {
        "model": "qwen/qwen3-vl-235b-a22b-instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=180)
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


def extract_pdf_content(pdf_path):
    """提取PDF中的所有图像和文本内容"""
    try:
        doc = fitz.open(pdf_path)
        images = []
        full_text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]

            # 提取文本
            page_text = page.get_text()
            full_text += f"\n=== 第 {page_num + 1} 页 ===\n{page_text}\n"

            # 提取图像
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


def extract_fss_parameters_from_text(full_text):
    """使用API从PDF文本中提取FSS参数（X、Y、Z）"""
    print("\n" + "=" * 50)
    print("正在从PDF文本中提取FSS参数...")
    print("=" * 50)

    parameter_extraction_prompt = f"""请分析提供的PDF文档文本，提取FSS单元的尺寸参数。

任务要求：
1. 提取FSS单元的三个关键尺寸：
   - X：单元在X方向的长度（最外环）
   - Y：单元在Y方向的长度（最外环）
   - Z：单元的厚度（介质板厚度）

2. 参数识别规则：
   - X和Y（长宽）：
     * 首先提取论文的文本和表格中所有关于长度的参数（单位为mm或者μm）(比如 SW、L、a、b、c、w、g等参数） 
     * 接着尝试提取最外环的参数
     * 接着判断最外环的参数中是否出现Dx或者Dy，若出现Dx或者Dy，则直接默认Dx=X，Dy=Y，进行输出
     * 接着判断参数中是否出现f以及L1、L2，若参数中出现f以及L1、L2，则禁止将参数f以及L1、L2的值输出为X和Y
     * 若参数中没有出现Dx和Dy以及f，则将提取出的参数的数值最大的值作为X和Y
     * 注意将所有单位统一为mm进行比较
     * 必须找到X和Y的具体数值，严格禁止输出"unknown"
     * 如果未找到合适的X与Y值，则继续深入分析PDF中的所有文字和表格，包括图片标注、表格数据和文本描述
     * 务必穷尽所有可能的线索，直到找出具体的数值
     * 在这个任务中，没有找到具体的X和Y值是不可接受的结果

   - Z（厚度）：
     * 关键词：substrate thickness、dielectric thickness、PCB thickness、h等
     * 通常在文中会直接说明介质板厚度
     * 如果表格中出现d、h等表示厚度的参数，则提取其值作为Z
     * 如果文中没有提到介质板厚度，则默认输出Z为1.6mm

3. 数据来源优先级：
   - 优先查找所有带单位的长度参数
   - 确保单位统一（通常为mm）
   - 如有必要，将μm转换为mm
   - 认真分析文本中提到的参数间的关系，确定哪些参数表示FSS的外环尺寸

4. 参数分析策略：
   - 首先全面列出文中所有带单位的长度参数
   - 分析每个参数在文中的含义和用途
   - 找出最可能代表X和Y的参数（通常是周期相关参数或结构整体尺寸）
   - 如果有多个参数，需根据上下文判断哪个最合理

5. 返回格式（严格Python字典）：
fss_size = {{
    'X': '',  # 必须提供具体数值，禁止返回"unknown"
    'Y': '',  # 必须提供具体数值，禁止返回"unknown"
    'Z': '1.6',  # 如果未找到则使用默认值
    'Unit': 'mm'
}}


PDF文本内容：
{full_text[:20000]}

注意事项：
- 全面分析文本中提到的所有长度参数，记录它们的单位和数值
- 务必全文仔细搜索，确保不遗漏任何可能的参数
- 对于X和Y，必须基于文档中的实际数据提供具体值
- 严格禁止在X和Y字段中返回"unknown"或任何默认值
- 必须基于PDF内容本身提取到的数据进行判断
- Unit字段记录统一单位（mm）
- 数值只保留数字部分，不要包含单位"""

    api_result = call_openrouter_api(
        parameter_extraction_prompt,
        image_data=None,
        max_tokens=60000
    )

    if not api_result:
        print("API参数提取失败")
        return {}

    try:
        # 解析API返回结果
        fss_size_match = re.search(r"fss_size\s*=\s*(\{.*?})", api_result, re.DOTALL)
        if fss_size_match:
            fss_size = eval(fss_size_match.group(1))
            # 验证必须包含X、Y、Z、Unit字段
            required_fields = ['X', 'Y', 'Z', 'Unit']
            if all(field in fss_size for field in required_fields):
                # 检查X和Y是否有具体数值
                if not fss_size['X'] or fss_size['X'] == 'unknown':
                    print("警告：API未能找到X值，需要重试")
                    # 这里我们不设默认值，而是返回空字典，表示需要重试
                    return {}
                if not fss_size['Y'] or fss_size['Y'] == 'unknown':
                    print("警告：API未能找到Y值，需要重试")
                    # 这里我们不设默认值，而是返回空字典，表示需要重试
                    return {}
                return fss_size
            else:
                print(f"警告：返回字典缺少必需字段，当前字段：{list(fss_size.keys())}")
                # 不设置默认值，返回空字典表示需要重试
                return {}
        else:
            print("未找到fss_size字典")
            print(f"API返回内容：\n{api_result}")
            # 返回空字典表示需要重试
            return {}
    except Exception as e:
        print(f"参数解析错误: {e}")
        print(f"完整API返回：\n{api_result}")
        # 返回空字典表示需要重试
        return {}


def select_fss_structure_image(encoded_images):
    """使用API逐个判断并选择FSS结构图"""
    select_prompt = """请判断这张图像是否为FSS结构图。

判断标准:
✓ 必须包含：箭头、英文长度参数（如 SW、D、L、W、P、a、b、g、h、r等）
✓ 必须显示：几何结构（如方形环、十字形、圆环等）

✗ 禁止单独包含：
  - 频率标注（GHz）
  - dB值标注
  - 曲线图（S参数图、频率响应图等）
  - A/m柱状图
  - 任何图表、坐标轴

特别注意：如果一个图片中既包含箭头、英文长度参数、几何结构以及其它内容（如曲线图、柱状图等），那么也认为是符合要求的。例如：一张图片里包含(a)、(b)、(c)...的子图，其中(a)是结构图，但是(b)、(c)是电流分布图，这种情况也应判定为有效结构图。

✗ 禁止提取没有任何英文长度参数以及几何结构（如方形环、十字形、圆环等）的图片

返回格式（严格JSON）：
{
    "is_fss_structure": true/false,
    "reason": "判断理由（简短说明）"
}"""

    for img_info in encoded_images:
        print(f"\n正在判断图像：页{img_info['page_num']}, 索引{img_info['img_index']}")

        api_result = call_openrouter_api(select_prompt, img_info["base64"])

        if api_result:
            try:
                # 尝试提取JSON
                json_match = re.search(r'\{.*\}', api_result, re.DOTALL)
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


def analyze_fss_pdf(pdf_path, output_folder = r"fss_out"):
    """完全基于API的FSS PDF文档综合分析（调整顺序：先文本分析，再图片处理，最后颜色材料分析）"""

    # 1. 提取PDF中的所有图像和文本
    all_images, full_text = extract_pdf_content(pdf_path)
    if not all_images:
        print("无法从PDF中提取任何图像")
        return {}, {}, {}, {}

    print(f"\n成功提取 {len(all_images)} 张图像")
    print(f"提取文本长度: {len(full_text)} 字符")

    # 2. 先进行文本分析，提取FSS周期参数（X、Y、Z）
    max_attempts = 3
    fss_size = {}
    for attempt in range(max_attempts):
        fss_size = extract_fss_parameters_from_text(full_text)
        if fss_size and 'X' in fss_size and 'Y' in fss_size and fss_size['X'] and fss_size['Y']:
            print(f"成功提取FSS周期参数（尝试 {attempt + 1}/{max_attempts}）")
            break
        print(f"提取FSS周期参数失败，尝试 {attempt + 1}/{max_attempts}")

    if not fss_size or not fss_size.get('X') or not fss_size.get('Y'):
        print("警告：经过多次尝试，仍未能提取FSS周期参数")
        return {}, {}

    print("\n" + "=" * 50)
    print("FSS周期参数提取完成")
    print("=" * 50)
    print(f"提取结果: {json.dumps(fss_size, indent=2, ensure_ascii=False)}")

    # 3. 转换图像为base64
    encoded_images = []
    for img_info in all_images:
        encoded_img = encode_image_to_base64(img_info["image"])
        encoded_images.append(
            {
                "base64": encoded_img,
                "page_num": img_info["page_num"],
                "img_index": img_info["img_index"],
                "image": img_info["image"],
            }
        )

    # 4. 使用API逐个判断并选择FSS结构图
    print("\n" + "=" * 50)
    print("开始逐个判断图像...")
    print("=" * 50)

    selected_image = select_fss_structure_image(encoded_images)

    if not selected_image:
        print("\n!!! 错误：该PDF中未找到符合标准的FSS结构图")
        return {}, {}, fss_size

    # 5. 保存选中的FSS结构图
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder,
        f"fss_structure_p{selected_image['page_num']}_i{selected_image['img_index']}.png",
    )
    pic_name = f"fss_structure_p{selected_image['page_num']}_i{selected_image['img_index']}.png"
    selected_image["image"].save(output_path)
    print(f"\n已保存FSS结构图: {output_path}")

    # 6. 最后调用API进行颜色和材料分析
    print("\n" + "=" * 50)
    print("正在调用API分析FSS结构图颜色和材料...")
    print("=" * 50)

    color_material_analysis_prompt = """请对提供的FSS (Frequency Selective Surface) 结构图进行颜色和材料分析。

请注意:
- 金属部分通常对应颜色较深的区域（如黑色、深灰色、蓝色等）→ 标记为'PEC'
- 介质部分通常对应颜色较浅的区域（如白色、浅灰色等）→ 标记为'FR4'
- 确保分析所有可见颜色，不要遗漏任何颜色
- 不要提取坐标轴的颜色，只提取FSS结构本身的颜色

必须返回Python字典格式（可直接使用），键名和结构不可修改：

col_mats = {{
    'black': 'PEC',
    'white': 'FR4',
    'red': 'PEC',
    'blue': 'FR4',
    # 等等，包含所有识别到的颜色
}}

注意：
- 返回格式必须严格遵循上述字典格式，以便可以直接在Python中使用
- 材料类型通常是'PEC'或'FR4'
- 颜色名称应使用标准颜色名称（如black、white、red、green、blue等）
- 即使颜色很少，也要确保分析每一种可见的颜色"""

    api_result = call_openrouter_api(
        color_material_analysis_prompt,
        selected_image["base64"],
        max_tokens=6000
    )

    if not api_result:
        print("API颜色材料分析失败")
        return {}, fss_size

    # 7. 解析API返回结果
    try:
        print("\nAPI返回结果预览：")
        print(api_result[:500])
        print("\n解析结果...")

        col_mats_match = re.search(r"col_mats\s*=\s*(\{.*?\})", api_result, re.DOTALL)
        col_mats = eval(col_mats_match.group(1)) if col_mats_match else {}

        return col_mats, fss_size, pic_name
    except Exception as e:
        print(f"API结果解析错误: {e}")
        print(f"完整API返回：\n{api_result}")
        return {}, fss_size


# 测试代码
if __name__ == "__main__":
    test_pdf_path = r"C:\Users\Administrator\Desktop\FSS论文\滤波：Design of a Miniaturized and Polarization-Independent Frequency-Selective Surface for Targeted EMI Shielding.pdf"
    col_mats, fss_size, pic = analyze_fss_pdf(test_pdf_path)

    print("\n" + "=" * 50)
    print("FSS PDF分析结果")
    print("=" * 50)
    print(f"1. 颜色-材料映射 (col_mats):\n{json.dumps(col_mats, indent=2, ensure_ascii=False)}")
    print(f"2. FSS周期尺寸 (fss_size):\n{json.dumps(fss_size, indent=2, ensure_ascii=False)}")
