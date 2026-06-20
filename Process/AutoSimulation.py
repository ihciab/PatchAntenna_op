import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesian_optimization.simulation.cst_library_path import ensure_cst_library_path

ensure_cst_library_path()

import cst.results
from Simulink.Simulation import *
import Rebuild.Fss_analyzer as Fa
import Rebuild.FSSfigDetector as Fc

class Solve(Simulation):
    def __init__(self, image_path, folder_path, proj_name, col_mats, fss_package: dict, color_ranges=None):
        super(Solve, self).__init__(image_path, folder_path, proj_name, col_mats, fss_package, color_ranges)

    def forward_process(self, save = True):
        self.start_simulation()
        if self._Solver:
            print('save model')
            export_image = ch.cst_export_pic(self._Folder_path)
            self._Modeler.add_to_history('ExportImage', export_image)
            ch.cst_close_project(self._De, self._Cst_proj)
            print('save S-parameters')
            project = cst.results.ProjectFile(self._Folder_path + '\\' + self._Project_name)
            s11 = project.get_3d().get_result_item(r"1D Results\S-Parameters\SZmax(1),Zmax(1)").get_data()
            s12 = project.get_3d().get_result_item(r"1D Results\S-Parameters\SZmax(1),Zmin(1)").get_data()
            # 列表中的第一个元组元素代表S参数（S - Parameter）的频率。
            # 第二个元组元素代表复数值的S参数。
            # 第三个元组元素代表S参数的复数值参考阻抗（Reference Impedance）。
            # a = project.get_3d().get_tree_items()
            with open(self._Folder_path + '\\s11.txt', 'w', encoding='utf-8') as f:
                print(s11, file=f)
            with open(self._Folder_path + '\\s12.txt', 'w', encoding='utf-8') as f:
                print(s12, file=f)

def main(paper_path, folder_path, proj_name, f0, f1, col_mats1, fss):
    col_mats, fss_size, pic = Fa.analyze_fss_pdf(paper_path, output_folder=fr"{folder_path}\fss_out")
    if fss_size["X"] == "unknown":
        fss_size["X"] = 6.6
    if fss_size["Y"] == "unknown":
        fss_size["Y"] = 6.6
    if fss_size["Z"] == "unknown":
        fss_size["Z"] = 1.6
    fss_size.update({'f0': f0, 'f1': f1, 'ground': False})
    fss_size["X"] = fss["X"]
    fss_size["Y"] = fss["Y"]
    fss_size["Z"] = fss["Z"]

    detector = Fc.FSSfigDetector(max_k=6)
    print(fr"{folder_path}\fss_out\{pic}")
    results = detector.detect(fr"{folder_path}\fss_out\{pic}", output_folder=fr'{folder_path}\fss_clear')
    os.makedirs(folder_path, exist_ok=True)

    sol = Solve(fr'{folder_path}\fss_clear\repair_fig.png', folder_path, proj_name, col_mats1, fss_size)
    sol.forward_process(save=True)

    flag = input('是否开始扫参(Y/n)')
    if flag == 'Y' or flag == 'y':
        epoch = input('一共进行扫参几轮')
        print(f'开始扫参，共{epoch}轮')
        sol.sweep_process(epoch=int(epoch))
    else:
        print('完成求解')
        input('按任意键退出...')


