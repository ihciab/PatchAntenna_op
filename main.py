import shutil
import sys
import json
import os

try:
    with open("config.json", "r", encoding="utf-8") as f:
        config_data = json.load(f)
# except FileNotFoundError:
#     print("错误：JSON文件不存在")
#     input("1")
except json.JSONDecodeError:
    print("错误：JSON格式无效（可能有语法错误）")
except Exception as e:
    print(f"读取失败：{e}")
else:
    # 无错误时执行
    print("读取成功：", config_data)
    try:
        sys.path.append(config_data["cstModuleBase"]["absolute_path"])
        import cst
    except ModuleNotFoundError:
        print('未找到python_cst_libraries文件')
        cst_lib_path = input(
            '请输入用户python_cst_libraries文件的绝对地址：')  # r"D:\CST2023\AMD64\python_cst_libraries"
        try:
            sys.path.append(cst_lib_path)
            import cst
        except ModuleNotFoundError:
            print('未找到python_cst_libraries文件')
            input('按任意键退出...')
        else:
            print("更新配置文件")
            config_data["cstModuleBase"]["absolute_path"] = cst_lib_path
            with open("config.json", "w", encoding="utf-8") as f:
                # indent=2：保持JSON格式缩进（美观）；ensure_ascii=False：正确处理中文
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            print("已导入cst模块")



    if config_data["file_handler"]["keep_project"] == "False":
        keep_project = False
    elif config_data["file_handler"]["keep_project"] == "True":
        keep_project = True
    else:
        keep_project = False


paper_dict={"Analysis_of_Scanning_Error_of_Array_Antenna_by_FSS_Radome.pdf": {
    "Col_mats":{
        "white": "FR4",
        "green": "PEC"
    },
    "FSS_package": {"X": 23, "Y": 23, "Z": 0.5, "f0": 0.5, "f1": 3}
},
    "A_Miniaturized_Dual-Band_FSS_With_Controllable_Frequency_Resonances.pdf": {
        "Col_mats": {
            "white": "FR4",
            "gray": "PEC"
        },
        "FSS_package": {"X": 6.6, "Y": 6.6, "Z": 1, "f0": 0.5, "f1": 11}
    },
    "A_FSS-_based_Single_Layer_Reflective_Polarizer_for_X_Ku-_Bands.pdf": {
        "Col_mats": {
            "red": "PEC",
            "orange": "FR4"
        },
        "FSS_package": {"X": 8.8, "Y": 8.8, "Z": 0.5, "f0": 0.5, "f1": 18}
    }}


import Process.SweepFSS
import Process.AutoSimulation

class UIAutoFss:
    def __init__(self):
        print('##########FSS自动仿真平台##########')
        print('请输入论文的绝对地址，多篇论文的地址请用逗号隔开')
        image_path = input('(如果要看演示请输入show)：')

        if image_path == 'show':
            Process.SweepFSS.main()
        else:
            paper_list = image_path.split(',')
            # folder_path = input('输入工程目录：')
            folder_path = config_data["file_handler"]["cst_proj"]

            if not keep_project:
                if os.path.exists(folder_path):
                    # 判断是否为文件夹（避免误删文件）
                    if os.path.isdir(folder_path):
                        # 删除文件夹（包括内部所有文件和子文件夹）
                        shutil.rmtree(folder_path)
                        print(f"文件夹 '{folder_path}' 已删除")
                    else:
                        print(f"路径 '{folder_path}' 存在但不是文件夹，无法删除")

            for i, paper in enumerate(paper_list):
                print(paper)
                temp_paper = paper.split("\\")
                print(type(temp_paper[-1]))
                paper_path = paper
                folder_path_temp = folder_path + fr"\{i}"
                proj_name = 'FSS.cst'
                # f0, f1 = input('输入频率范围f0<f1,用空格隔开').split(' ')
                f0, f1 = paper_dict[temp_paper[-1]]["FSS_package"]["f0"], paper_dict[temp_paper[-1]]["FSS_package"]["f1"]
                Process.AutoSimulation.main(paper_path, folder_path_temp, proj_name, float(f0), float(f1),
                                            col_mats1=paper_dict[temp_paper[-1]]["Col_mats"],
                                            fss=paper_dict[temp_paper[-1]]["FSS_package"])


if __name__ == '__main__':
    StartUI = UIAutoFss()