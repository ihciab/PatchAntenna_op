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

def main():
    paper_path = r"Analysis_of_Scanning_Error_of_Array_Antenna_by_FSS_Radome.pdf"
    Folder_path = r'testFss'
    Proj_name = 'testFss.cst'

    col_mats, fss_size, pic = Fa.analyze_fss_pdf(paper_path, output_folder=fr"{Folder_path}\fss_out")
    fss_size.update({'f0': 0.5, 'f1': 3, 'ground': False})

    detector = Fc.FSSfigDetector(max_k=6)
    print(fr"{Folder_path}\fss_out\{pic}")
    results = detector.detect(fr"{Folder_path}\fss_out\{pic}", output_folder=fr'{Folder_path}\fss_clear')
    os.makedirs(Folder_path, exist_ok=True)

    sol = Solve(fr'{Folder_path}\fss_clear\repair_fig.png', Folder_path, Proj_name, col_mats, fss_size)
    sol.forward_process(save=True)
    e = 2
    print(f'开始扫参，共{e}轮')
    sol.sweep_process(epoch=e)


if __name__ == '__main__':
    main()
