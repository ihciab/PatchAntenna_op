import os
import json
import math
import sys
from pathlib import Path

# Support running this file directly with:
# `python Simulink/Simulation.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REBUILD_DIR = PROJECT_ROOT / "Rebuild"
if str(REBUILD_DIR) not in sys.path:
    sys.path.insert(0, str(REBUILD_DIR))

import Simulink.handle as ch
from Rebuild.ImageInit import *
from Rebuild.BSplineContour import BSplineContour
from Rebuild.Curves2Component import Curves2Components
from Rebuild.Image2Pixel import Image2Pixel
from Rebuild.PortSearch import SubjectEdgeAnalyzer

from Sweep.IntersectionDetection import IntersectionDetection
from Sweep.ReshapeContour import ReshapeContour

import cst
import cst.results

print(cst.__file__)

class Simulation:
    def __init__(self, instance_dict, color_ranges=None):
        """
        Instance_dict = {
        'Folder_path': r"D:\CST2023proj\autocst\testFss2",
        'Instance': 'FSS',
        'mode': 'S',
        'Units': ['mm', 'GHz'],
        'FSS_package': {'X': 6.6, 'Y': 6.6, 'f0': 0.5, 'f1': 13},
        'layers': {
            'layer0': {'img_path': r"D:\pyproject\Auto_py2cst_v0.71\test\test22.png",
                       'substrate': 0.6,
                       'gnd': False,
                       'col_mats': {
                           'white': 'FR4',
                           'blue': 'PEC',
                       },
                       },
            'layer1': {'img_path': r"D:\pyproject\Auto_py2cst_v0.71\test\test23.png",
                       'substrate': 0.5,
                       'gnd': False,
                       'col_mats': {
                           'white': 'FR4',
                           'blue': 'PEC',
                       },
                       },
            'layer2': {'img_path': r"D:\pyproject\Auto_py2cst_v0.71\test\test24.png",
                       'substrate': 0.4,
                       'gnd': True,
                       'col_mats': {
                           'white': 'FR4',
                           'blue': 'PEC',
                       },
                       },
        },
    }
        """
        # 初始化图片
        self.__Mats = None
        self._Folder_path = instance_dict['Folder_path']
        self._Instance = instance_dict['Instance'] + '.cst'
        self.__Layers = instance_dict['layers']
        self.__Fss_package = instance_dict['FSS_package']
        self.__Color_ranges = color_ranges
        self._Modeler = None
        self._De = None
        self._Cst_proj = None
        self.__listOfContours_dict = None # contours_dict, curves_tree, solids_mask, col_mats, anchor
        self.newcons_dict = None
        self.__Solids_mask = None
        self.__Scale = None
        self.__Absolut_anchor = None
        self.__Height = None
        self.__Metal_thickness = float(instance_dict.get('Metal_thickness', 0.035))
        self._Solver = False
        self.IP = Image2Pixel()

    def rebuildfig(self):
        from Rebuild.FssDetector import FSSfigDetector

        ## 将 图片经过FSS处理后 修改 Instance_dict 里的图片路径
        detector = FSSfigDetector(max_k=6)
        image_path = "./fss_test/019.png"
        results = detector.detect(image_path)
        return self.__do_modeling()

    def __simulation_process(self):
        self.__do_modeling()

        if os.path.exists('port_summary.json'):
            with open('port_summary.json', 'r', encoding='utf-8') as file:
                ports = json.load(file)
        if ports.get("border_contact_mode") == "separate":
            print(
                "Warning: detected PEC edge does not touch the image border; "
                "the auto-generated waveguide port may be unstable."
            )
        x1, y1 = ports["closest_edge"][0]
        x2, y2 = ports["closest_edge"][1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        orientation = self.__resolve_port_orientation(ports, x1, y1, x2, y2)
        if (x2 - x1) < (y2 - y1):
            set_port = ch.cst_waveguide_port(orientation,
                                             math.floor(x1 * self.__Scale),
                                             math.ceil(x2 * self.__Scale),
                                             math.floor(y1 * self.__Scale) - self.__Height * 3,
                                             math.ceil(y2 * self.__Scale) + self.__Height * 3,
                                             -self.__Height * 1.5, self.__Height * 2.5)
            self._Modeler.add_to_history(f'set port', set_port)
        else:
            set_port = ch.cst_waveguide_port(orientation,
                                             math.floor(x1 * self.__Scale) - self.__Height * 3,
                                             math.ceil(x2 * self.__Scale) + self.__Height * 3,
                                             math.floor(y1 * self.__Scale),
                                             math.ceil(y2 * self.__Scale),
                                             -self.__Height * 1.5, self.__Height * 2.5)
            self._Modeler.add_to_history(f'set port', set_port)

        self.__start_solve(self._Folder_path, self._Instance)

    def __sweeping(self, show=True, epoch:int=2):
        for i in range(epoch):
            self._Modeler, self._De, self._Cst_proj = self.__init_cst(self._Folder_path, f'{i}.cst')
            self.__Mats = None
            height = 0
            for j, layer in enumerate(self.__listOfContours_dict):
                new_cons_dict = layer[0]
                curves_tree = layer[1]
                solids_mask = layer[2]
                col_mats = layer[3]
                anchor = layer[4]
                iD = IntersectionDetection()

                for contour in new_cons_dict:
                    print(f'寻找曲线{contour}')
                    if contour == '1':
                        rc = ReshapeContour(new_cons_dict[contour]['contour'].reshape(-1, 2).copy(),
                                            new_cons_dict[contour]['loss']['step'])
                        new_con = rc.reshape_contour(show=show)
                        flag = iD.find_intersecting_pairs(new_con)
                        # print(flag)
                        while not flag:
                            rc = ReshapeContour(new_cons_dict[contour]['contour'].reshape(-1, 2).copy(),
                                                new_cons_dict[contour]['loss']['step'])
                            new_con = rc.reshape_contour(show=show)
                            flag = iD.find_intersecting_pairs(new_con)
                            # print(flag)
                        print(f'已完成{j + 1}组扫参, 还剩{epoch - j - 1}组')
                        new_cons_dict[contour]['fitting']['points'] = new_con

                height = self.__modeling(new_cons_dict, curves_tree, solids_mask, col_mats, anchor, j, height)

            self.__start_solve(self._Folder_path, f'{i}.cst')

    def __do_modeling(self):
        self._Modeler, self._De, self._Cst_proj = self.__init_cst()
        # 创建Modeler、 De、 Cst_proj这些是对cst操作的基本接口

        if ((self._Modeler is None) or
                (self._De is None) or
                (self._Cst_proj is None)):
            raise RuntimeError("CST模型器未初始化，请先完成建模")

        height = 0
        for i, layer in enumerate(self.__Layers):
            img_path = self.__Layers[layer]['img_path']
            col_mats = self.__Layers[layer]['col_mats']
##############################修改图片路径###############
            for cols in col_mats:
                if col_mats[cols] == 'PEC':
                    print('writing')
                    self.__get_port(image_path=img_path, subject_color=cols, name=layer+'2PEC')

            contours_dict, curves_tree, solids_mask, anchor = self.__init_datas(img_path, col_mats)

            if self.__listOfContours_dict is None:
                self.__listOfContours_dict = []
                self.__listOfContours_dict.append([contours_dict, curves_tree, solids_mask, col_mats, anchor])
            else:
                self.__listOfContours_dict.append([contours_dict, curves_tree, solids_mask, col_mats, anchor])
            if self.__Absolut_anchor is None:
                self.__Absolut_anchor = anchor

            height = self.__modeling(contours_dict, curves_tree, solids_mask, col_mats, anchor, i, height)
        self.__Height = height

    def __init_cst(self, folder_path=None, project_name=None):
        if folder_path is None:
            folder_path = self._Folder_path
        if project_name is None:
            project_name = self._Instance

        if os.path.exists(folder_path):
            pass
        else:
            os.makedirs(folder_path)
        # 创建一个新的工程，进一下create（对一个新的工程的，要先创建工程，对已经存在的工程就不需要创建直接open）
        ch.cst_create_project(folder_path + '\\' + project_name)
        # 打开工程环境
        de, cst_prj, _ = ch.cst_open_project(folder_path + '\\' + project_name)
        # 取出模型
        modeler = cst_prj.modeler

        # 进入自动化脚本的初始化流程，进一下init
        ch.cst_auto_init(modeler, self.__Fss_package['f0'], self.__Fss_package['f1'])

        return modeler, de, cst_prj

    def __load_material(self, col_mats):
        if self.__Mats is None:
            self.__Mats = []
        for col in col_mats:
            if (col_mats[col] != 'PEC' and 'Vacuum' and 'unknown') and (col_mats[col] not in self.__Mats):
                self.__Mats.append(col_mats[col])
                lod_mat = ch.cst_load_material(col_mats[col])
                self._Modeler.add_to_history(f'load {col_mats[col]}', lod_mat)

    def __init_datas(self, image_path, col_mats, show=True):
        # 对图形进行基本处理
        ii = ImageInit(image_path, show=show, save=self._Folder_path)
        img = ii.centered_img()
        edge = ii.edges()
        anchor_point = ii.center_point()

        # 对轮廓进行B样条拟合
        bc = BSplineContour(img, edge, show=show, save=self._Folder_path)
        contours_dict = bc.get_contours_dict()
        curves_tree = bc.get_curves_tree()
        print(curves_tree)

        # 根据轮廓拟合曲线的父子类关系确定solid的材料
        c2c = Curves2Components(img, img.shape, contours_dict, curves_tree, self.__Color_ranges)
        solids_mask = c2c.solids_col()
        self.IP.image2pixel(img,
                            self._Folder_path,
                            curves_tree,
                            contours_dict,
                            solids_mask,
                            col_mats)
        return contours_dict, curves_tree, solids_mask, anchor_point

    def __resolve_port_orientation(self, ports, x1, y1, x2, y2):
        border_sides = ports.get("closest_border_sides", [])
        axis = 'x' if (x2 - x1) < (y2 - y1) else 'y'
        orientation_map = {
            ('x', 'left'): 'xmin',
            ('x', 'right'): 'xmax',
            ('y', 'top'): 'ymin',
            ('y', 'bottom'): 'ymax',
        }

        for side in border_sides:
            orientation = orientation_map.get((axis, side))
            if orientation is not None:
                return orientation

        return f'{axis}max'

    def __nodes2curves2solid(self, cons_dict, layer, height, ab_anchor=(0, 0), re_anchor=(0, 0)):
        for curve in cons_dict:
            contour = cons_dict[curve]['fitting']['points']
            draw_curve = ch.cst_curves(name=curve, curve=layer, contour=contour)
            self._Modeler.add_to_history(f'draw_{layer}_{curve}', draw_curve)
            draw_solid = ch.cst_extrudecurve(name=curve,
                                             curve=layer+':'+curve,
                                             component=layer,
                                             material='PEC',
                                             thickness=self.__Metal_thickness
                                             )
            self._Modeler.add_to_history(f'set_{layer}_{curve}', draw_solid)
            if height > 0:
                trans_solid = ch.cst_translate(f'{layer}:{curve}',
                                               ab_anchor[0] - re_anchor[0],
                                               ab_anchor[1] - re_anchor[1],
                                               -float(height))
                self._Modeler.add_to_history(f'trans_{layer}:{curve}', trans_solid)

    def __solids2solids(self, cur_tree, layer, col_mats, sol_mask, height, ab_anchor=(0, 0), re_anchor=(0, 0)):
        for i, super_curve in enumerate(cur_tree):
            nn = super_curve
            for curve in cur_tree[super_curve]:
                nn += ('-' + curve)
                sub = ch.cst_del_subtract(layer, super_curve,
                                          layer, curve)
                self._Modeler.add_to_history(f'subtract_{layer}_{nn}', sub)
                draw_solid = ch.cst_extrudecurve(name=curve,
                                                 curve=layer + ':' + curve,
                                                 component=layer,
                                                 material='PEC',
                                                 thickness=self.__Metal_thickness
                                                 )
                self._Modeler.add_to_history(f'set_{layer}_{curve}', draw_solid)
                if height > 0:
                    trans_solid = ch.cst_translate(f'{layer}:{curve}',
                                                   ab_anchor[0] - re_anchor[0],
                                                   ab_anchor[1] - re_anchor[1],
                                                   -float(height))
                    self._Modeler.add_to_history(f'trans_{layer}:{curve}', trans_solid)
            rn = ch.cst_solid_rename(layer, super_curve, new_name=nn)
            print('nn', nn)
            self._Modeler.add_to_history(f'SolidRename_{nn}', rn)
            if col_mats[sol_mask[nn]] != 'PEC':
                ds = ch.cst_del_solid(layer, nn)
                self._Modeler.add_to_history(f'del_solid_{nn}', ds)

    def __get_scale(self, con_dict, curs_tree, layer, height,
                    ab_anchor=(0, 0), re_anchor=(0, 0), substrate:float=0, gnd:bool=False):
        for i, super_curve in enumerate(curs_tree):
            if i > 0:
                break
            else:
                if self.__Scale is None:
                    self.__Scale = 2 * (float(self.__Fss_package['X']) + float(self.__Fss_package['Y'])) / int(
                        cv2.arcLength(con_dict[super_curve]['contour'], True))
                    print('scale', self.__Scale)
                if substrate > 0:
                    draw_solid = ch.cst_extrudecurve(name='sub_'+super_curve,
                                                     curve=layer+':'+super_curve,
                                                     component=layer,
                                                     material='Rogers RT-duroid 5880 (loss free)',
                                                     thickness=substrate
                                                     )
                    self._Modeler.add_to_history(f'set_{layer}_sub_{super_curve}', draw_solid)
                    trans_sub = ch.cst_translate(f'{layer}:sub_{super_curve}',
                                                   ab_anchor[0] - re_anchor[0],
                                                   ab_anchor[1] - re_anchor[1],
                                                 -float(height))
                    self._Modeler.add_to_history(f'set_{layer}_gnd', trans_sub)
                if gnd:
                    draw_solid = ch.cst_extrudecurve(name='gnd',
                                                     curve=layer+':'+super_curve,
                                                     component=layer,
                                                     material='PEC',
                                                     thickness=self.__Metal_thickness
                                                     )
                    self._Modeler.add_to_history(f'set_{layer}_gnd', draw_solid)
                    trans_gnd = ch.cst_translate(f'{layer}:gnd',
                                                 ab_anchor[0] - re_anchor[0],
                                                 ab_anchor[1] - re_anchor[1],
                                                 -float(height + substrate))
                    self._Modeler.add_to_history(f'set_{layer}_gnd', trans_gnd)

    def __start_solve(self, folder_path, project_name):
        solver = self._Modeler.run_solver()
        if solver:
            print('扫惨求解完成')
            print('save model')
            export_image = ch.cst_export_pic(folder_path, project_name)
            self._Modeler.add_to_history(f'ExportImage{project_name}', export_image)
            ch.cst_close_project(self._De, self._Cst_proj)
            print('save S-parameters')
            project = cst.results.ProjectFile(folder_path + '\\' + project_name)
            s11 = project.get_3d().get_result_item(r"1D Results\S-Parameters\S1,1").get_data()
            # 列表中的第一个元组元素代表S参数（S - Parameter）的频率。
            # 第二个元组元素代表复数值的S参数。
            # 第三个元组元素代表S参数的复数值参考阻抗（Reference Impedance）。
            # a = project.get_3d().get_tree_items()
            with open(folder_path + f'\\{project_name}_s11.txt', 'w', encoding='utf-8') as f:
                print(s11, file=f)
        else:
            raise ValueError('求解失败请检查流程')

    def __modeling(self, new_cons_dict, curves_tree, solids_mask, col_mats, anchor, idx, height=0):
        self.__load_material(col_mats)
        self.__nodes2curves2solid(new_cons_dict, f'layer{idx}', height, self.__Absolut_anchor, anchor)
        print(f'layer{idx}:{solids_mask}')

        self.__solids2solids(curves_tree, f'layer{idx}', col_mats, solids_mask, height, self.__Absolut_anchor, anchor)

        self.__get_scale(new_cons_dict, curves_tree, f'layer{idx}', height,
                         self.__Absolut_anchor, anchor,
                         substrate=self.__Layers[f'layer{idx}']['substrate'], gnd=self.__Layers[f'layer{idx}']['gnd'])

        height += self.__Layers[f'layer{idx}']['substrate']

        s = ch.cst_scale(f'layer{idx}', self.__Scale)
        self._Modeler.add_to_history(f'scal layer{idx}', s)

        del_curve = ch.cst_del_curves(f'layer{idx}')
        self._Modeler.add_to_history(f'del curve layer{idx}', del_curve)
        print('height:', height)
        return height

    def __get_port(self, image_path, subject_color, name:str):
        analyzer = SubjectEdgeAnalyzer(min_component_area=500, approx_epsilon_ratio=0.0025)

        result = analyzer.analyze(image_path, subject_color=subject_color)
        analysis_path = f"{name}_analysis.png"
        analyzer.visualize(result, save_path=analysis_path, show=False)

        summary = {
            "name": name,
            "image_path": str(image_path),
            "subject_color": subject_color,
            "analysis_path": str(analysis_path),
            "bbox": list(result.subject_component.bbox),
            "closest_edge": result.closest_edge.tolist(),
            "closest_edge_index": result.closest_edge_index,
            "closest_border_sides": list(result.closest_border_sides),
            "distance_to_image_border": round(result.distance_to_image_border, 2),
            "mean_distance_to_image_border": round(result.mean_distance_to_image_border, 2),
            "border_overlap_length": round(result.border_overlap_length, 2),
            "border_contact_mode": result.border_contact_mode,
        }

        summary_path = Path("port_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def simulation_process(self):
        self.__simulation_process()

    def sweeping(self, show=True, epoch:int=2):
        self.__sweeping(show, epoch)

    def solid_mask(self):
        return self.__Solids_mask

    def scale(self):
        return self.__Scale

    @property
    def modeler(self):
        return self._Modeler


if __name__ == '__main__':
    Instance_dict = {
        'Folder_path': r"D:\CST2023proj\autocst_MA\101010",
        'Instance': 'Microstrip_Antenna',
        'mode': 'S',
        'Units': ['mm', 'GHz'],
        'FSS_package': {'X': 36, 'Y': 36, 'f0': 6, 'f1': 14},
        'layers': {
            'layer0': {'img_path': r"D:\cst2py_box\Auto_py2cst_v0.71\test\test47.png",
                       'substrate': 0.6,
                       'gnd': True,
                       'col_mats': {
                           'white': 'Rogers RT-duroid 5880 (loss free)',
                           'gray': 'PEC',
                       },
                       },
        },
    }
    Instance_dict1 = {
        'Folder_path': r"D:\CST2023proj\autocst_MA\999",
        'Instance': 'Microstrip_Antenna',
        'mode': 'S',
        'Units': ['mm', 'GHz'],
        ####################################    传入原始图片 的尺寸和仿真频率范围，进行扫参优化    ####################################
        'FSS_package': {'X': 36, 'Y': 36, 'f0': 6, 'f1': 14},
        'layers': {
            'layer0': {'img_path': r"D:\cst2py_box\Auto_py2cst_v0.71\test\test48.png",
                       'substrate': 0.6,
                       'gnd': True,
                       'col_mats': {
                           'cyan': 'Rogers RT-duroid 5880 (loss free)',
                           'orange': 'PEC',
                       },
                       },
        },
    }
    simulation = Simulation(Instance_dict)
    simulation.simulation_process()

