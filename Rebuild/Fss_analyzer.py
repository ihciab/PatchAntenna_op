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


def call_aliyun_api(prompt, image_data=None, max_tokens=4000, temperature=0.1):
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

    # data = {
    #     "model": "qwen/qwen-2.5-vl-72b-instruct",
    #     "messages": messages,
    #     "max_tokens": max_tokens,
    #     "temperature": temperature,
    # }

    data = {
        "model": "qwen-vl-plus",  # 使用合适的阿里云视觉语言模型
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
    """使用API从PDF文本中提取FSS周期参数（X、Y、Z）"""
    print("\n" + "=" * 50)
    print("正在从PDF文本中提取FSS周期参数...")
    print("=" * 50)

    parameter_extraction_prompt = f"""请分析提供的PDF文档文本，提取FSS单元的周期尺寸参数。

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

    api_result = call_aliyun_api(
        parameter_extraction_prompt,
        image_data=None,
        max_tokens=6000
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


def select_fss_structure_image(encoded_images):
    """使用API逐个判断并选择FSS结构图"""
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

    for img_info in encoded_images:
        print(f"\n正在判断图像：页{img_info['page_num']}, 索引{img_info['img_index']}")

        api_result = call_aliyun_api(select_prompt, img_info["base64"])

        if api_result:
            try:
                # 尝试提取JSON
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
    fss_size = extract_fss_parameters_from_text(full_text)

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
        return {}, {}, {}, fss_size

    # 5. 保存选中的FSS结构图
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder,
        f"fss_structure_p{selected_image['page_num']}_i{selected_image['img_index']}.png",
    )
    pic_name = f"fss_structure_p{selected_image['page_num']}_i{selected_image['img_index']}.png"
    selected_image["image"].save(output_path)
    print(f"\n已保存FSS结构图: {output_path}")

    # 6. 生成标准颜色范围字符串
    standard_colors_str = json.dumps(STANDARD_COLOR_RANGES, indent=2, ensure_ascii=False)

    # 7. 最后调用API进行颜色和材料分析
    print("\n" + "=" * 50)
    print("正在调用API分析FSS结构图颜色和材料...")
    print("=" * 50)

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

    api_result = call_aliyun_api(
        color_material_analysis_prompt,
        selected_image["base64"],
        max_tokens=6000
    )

    if not api_result:
        print("API颜色材料分析失败")
        return {}, {}, {}, fss_size

    # 8. 解析API返回结果
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

        # return color_ranges, col_mats, col_comps, fss_size, pic_name
        return col_mats, fss_size, pic_name
    except Exception as e:
        print(f"API结果解析错误: {e}")
        print(f"完整API返回：\n{api_result}")
        return {}, {}, {}, fss_size


# 测试代码
if __name__ == "__main__":
    test_pdf_path = r"D:\pyproject\Auto_py2cst_v0.5\0.5版可演示论文\自己的论文\Analysis_of_Reconfigurable_Frequency_Selective_Surface_FSS_Using_Square_Closed_Split_Rings.pdf"

    col_mats, fss_size, pic = analyze_fss_pdf(test_pdf_path)

    print("\n" + "=" * 50)
    print("FSS PDF分析结果")
    print("=" * 50)
    print(pic)
    # print(f"1. 颜色范围 (color_ranges):\n{json.dumps(color_ranges, indent=2, ensure_ascii=False)}")
    print(f"\n2. 颜色-材料映射 (col_mats):\n{json.dumps(col_mats, indent=2, ensure_ascii=False)}")
    # print(f"\n3. 颜色-组件映射 (col_comps):\n{json.dumps(col_comps, indent=2, ensure_ascii=False)}")
    print(f"\n4. FSS周期尺寸 (fss_size):\n{json.dumps(fss_size, indent=2, ensure_ascii=False)}")
