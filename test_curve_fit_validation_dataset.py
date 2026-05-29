"""
基于验证集的测试脚本
专门用于测试 validation_dataset 中的验证样本

参照 test_curve_fit_metamaterial_antenna.py 的结构，但专门针对验证集设计
在可视化中使用方法全称（中文）
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist

# Windows控制台UTF-8支持
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from core.image.initializer import ImageInitializer as ImageInit
from core.geometry.bspline_fitter import BSplineContour
from core.geometry.nurbs_fitter import NURBSpineContour
from core.geometry.optimized_bspline_fitter import OptimizedBSplineFitter
from core.geometry.optimized_nurbs_fitter import OptimizedNURBSFitter
from core.geometry.curve_evaluator import CurveEvaluator
from core.geometry.geometry_aware_metrics import GeometryAwareMetrics, compute_geometry_aware_score
from core.geometry.segment_extractor import SegmentExtractor, extract_segments_from_fitted_contour


class ValidationDatasetTester:
    """验证集测试类"""
    
    # 方法全称映射（用于可视化）
    METHOD_FULL_NAMES = {
        'bspline': 'B样条拟合',
        'nurbs': 'NURBS拟合',
        'optimized_nurbs': '优化NURBS拟合',
        'optimized_bs': '优化B样条拟合',
    }
    
    def __init__(self, validation_dataset_dir: str = "validation_dataset",
                 output_dir: str = "validation_test_output",
                 save_json_report: bool = True):
        """
        初始化验证集测试器
        
        参数:
            validation_dataset_dir: 验证集目录路径
            output_dir: 输出目录
            save_json_report: 是否保存JSON报告
        """
        self.validation_dataset_dir = validation_dataset_dir
        self.output_dir = output_dir
        self.save_json_report = save_json_report
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载验证集摘要
        self.summary = self._load_summary()
        self.test_results = []
    
    def _load_summary(self) -> List[Dict]:
        """加载验证集摘要"""
        summary_path = os.path.join(self.validation_dataset_dir, "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"未找到验证集摘要文件: {summary_path}")
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"加载验证集摘要: {len(summary)} 个样本")
        return summary
    
    def _extract_boundary_gt_segments(self, centered_bgr_img: np.ndarray) -> List[Dict]:
        """从图像中提取边界GT分段"""
        if centered_bgr_img is None or centered_bgr_img.size == 0:
            return []

        if centered_bgr_img.ndim == 2:
            mask = (centered_bgr_img > 0).astype(np.uint8) * 255
        elif centered_bgr_img.ndim == 3 and centered_bgr_img.shape[2] == 3:
            mask = np.any(centered_bgr_img < 250, axis=2).astype(np.uint8) * 255
        else:
            return []

        if not np.any(mask):
            return []

        contours, _hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        gt_segments = []
        for c in contours:
            pts = c.reshape(-1, 2).astype(float)
            if len(pts) < 3:
                continue
            if len(pts) > 3000:
                step = int(np.ceil(len(pts) / 3000))
                pts = pts[::max(1, step)]
            try:
                gt_segments.extend(extract_segments_from_fitted_contour(pts))
            except Exception:
                continue

        return gt_segments
    
    def test_single_sample(self, sample: Dict, sample_index: int = 0) -> Dict:
        """
        测试单个验证样本
        
        参数:
            sample: 样本字典（从summary.json中加载）
            sample_index: 样本索引
        
        返回:
            测试结果字典
        """
        sample_name = sample.get('name', f'sample_{sample_index}')
        print(f"\n[{sample_index + 1}/{len(self.summary)}] 测试: {sample_name}")
        print("=" * 60)
        
        # 加载图像和GT
        image_path = sample.get('image_path', '')
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.validation_dataset_dir, image_path.replace('validation_dataset\\', ''))
        
        gt_path = sample.get('gt_path', '')
        if not os.path.isabs(gt_path):
            gt_path = os.path.join(self.validation_dataset_dir, gt_path.replace('validation_dataset\\', ''))
        
        if not os.path.exists(image_path):
            print(f"   [ERROR] 图像文件不存在: {image_path}")
            return {
                'sample_name': sample_name,
                'success': False,
                'error': f'图像文件不存在: {image_path}'
            }
        
        try:
            # 1. 图像初始化（使用原始边缘检测）
            print("1. 图像初始化...")
            ii = ImageInit(image_path, show=False, save="")
            img = ii.centered_img()
            edges = ii.edges()
            print(f"   [OK] 图像尺寸: {img.shape}")
            print(f"   [OK] 边缘图尺寸: {edges.shape}")
            
            # 提取边界GT分段
            gt_boundary_segments = self._extract_boundary_gt_segments(img)
            
            # 2. B样条轮廓拟合
            print("\n2. B样条轮廓拟合...")
            bc = BSplineContour(
                img=img,
                edges=edges,
                lr=0.1,
                delta=2,
                step=30,
                threshold=10.0,
                show=False,
                save=""
            )
            contours_dict_bs = bc.get_contours_dict()
            curves_tree_bs = bc.get_curves_tree()
            print(f"   [OK] 检测到 {len(contours_dict_bs)} 个轮廓")
            
            # 3. NURBS轮廓拟合
            print("\n3. NURBS轮廓拟合...")
            nbs = NURBSpineContour(
                img=img,
                edges=edges,
                lr=0.1,
                delta=2,
                step=30,
                threshold=10.0,
                degree=3,
                show=False,
                save=""
            )
            contours_dict_nurbs = nbs.get_contours_dict()
            curves_tree_nurbs = nbs.get_curves_tree()
            print(f"   [OK] 检测到 {len(contours_dict_nurbs)} 个轮廓")
            
            # 4. 优化NURBS拟合
            print("\n4. 优化NURBS拟合（曲率驱动权重+分段）...")
            segments_info_optimized_nurbs = {}
            try:
                optimized_nurbs = OptimizedNURBSFitter(
                    img=img,
                    edges=edges,
                    lr=0.1,
                    delta=2,
                    step=30,
                    threshold=10.0,
                    degree=3,
                    line_threshold=2.0,
                    arc_threshold=2.0,
                    curvature_threshold=0.15,
                    curvature_weight_alpha=5.0,
                    use_segmentation=True,
                    show=False,
                    save=""
                )
                contours_dict_optimized_nurbs = optimized_nurbs.get_contours_dict()
                curves_tree_optimized_nurbs = optimized_nurbs.get_curves_tree()
                segments_info_optimized_nurbs = optimized_nurbs.get_segments_info()
                print(f"   [OK] 检测到 {len(contours_dict_optimized_nurbs)} 个轮廓")
            except Exception as e:
                print(f"   [WARN] 优化NURBS拟合失败: {e}")
                contours_dict_optimized_nurbs = {}
                curves_tree_optimized_nurbs = {}
                segments_info_optimized_nurbs = {}
            
            # 5. 优化B样条拟合
            print("\n5. 优化B样条拟合（几何语义感知）...")
            try:
                optimized_bs_fitter = OptimizedBSplineFitter(
                    img=img,
                    edges=edges,
                    line_threshold=2.0,
                    arc_threshold=2.0,
                    curvature_threshold=0.15,
                    spline_degree=3,
                    show=False,
                    save=""
                )
                contours_dict_optimized_bs = optimized_bs_fitter.get_contours_dict()
                curves_tree_optimized_bs = optimized_bs_fitter.get_curves_tree()
                print(f"   [OK] 检测到 {len(contours_dict_optimized_bs)} 个轮廓")
            except Exception as e:
                print(f"   [WARN] 优化B样条拟合失败: {e}")
                contours_dict_optimized_bs = {}
                curves_tree_optimized_bs = {}
            
            # 6. 评估对比
            print("\n6. 精细评估对比...")
            evaluation_comparison = self._compare_fitting_methods(
                img, edges,
                contours_dict_bs, contours_dict_nurbs,
                contours_dict_optimized_nurbs, contours_dict_optimized_bs,
                segments_info_optimized_nurbs=segments_info_optimized_nurbs,
                gt_segments=gt_boundary_segments
            )
            
            bspline_score = evaluation_comparison.get('bspline', {}).get('overall', {}).get('overall', 0)
            nurbs_score = evaluation_comparison.get('nurbs', {}).get('overall', {}).get('overall', 0)
            optimized_nurbs_score = evaluation_comparison.get('optimized_nurbs', {}).get('overall', {}).get('overall', 0)
            optimized_bs_score = evaluation_comparison.get('optimized_bs', {}).get('overall', {}).get('overall', 0)
            
            print(f"   [OK] B样条拟合综合得分: {bspline_score:.2f}")
            print(f"   [OK] NURBS拟合综合得分: {nurbs_score:.2f}")
            print(f"   [OK] 优化NURBS拟合综合得分: {optimized_nurbs_score:.2f}")
            print(f"   [OK] 优化B样条拟合综合得分: {optimized_bs_score:.2f}")
            
            # 7. 生成可视化
            print("\n7. 生成可视化对比图...")
            vis_path = self._visualize_method_overview(
                processed_img=img,
                edges=edges,
                contours_dict_by_method={
                    'bspline': contours_dict_bs,
                    'nurbs': contours_dict_nurbs,
                    'optimized_nurbs': contours_dict_optimized_nurbs,
                    'optimized_bs': contours_dict_optimized_bs,
                },
                gt_segments=gt_boundary_segments,
                evaluation_comparison=evaluation_comparison,
                test_name=sample_name
            )
            print(f"   [OK] 可视化对比图已保存: {vis_path}")
            
            result = {
                'sample_name': sample_name,
                'success': True,
                'image_path': image_path,
                'gt_path': gt_path,
                'visualization_path': vis_path,
                'complexity_level': sample.get('complexity_level', -1),
                'structure_tags': sample.get('structure_tags', []),
                'bspline': {
                    'contour_count': len(contours_dict_bs),
                    'tree_nodes': len(curves_tree_bs),
                },
                'nurbs': {
                    'contour_count': len(contours_dict_nurbs),
                    'tree_nodes': len(curves_tree_nurbs),
                },
                'optimized_nurbs': {
                    'contour_count': len(contours_dict_optimized_nurbs) if contours_dict_optimized_nurbs else 0,
                    'tree_nodes': len(curves_tree_optimized_nurbs) if curves_tree_optimized_nurbs else 0,
                },
                'optimized_bs': {
                    'contour_count': len(contours_dict_optimized_bs) if contours_dict_optimized_bs else 0,
                    'tree_nodes': len(curves_tree_optimized_bs) if curves_tree_optimized_bs else 0,
                },
                'evaluation_comparison': evaluation_comparison,
            }
            
            print("\n[OK] 测试完成")
            return result
            
        except Exception as e:
            print(f"\n[FAIL] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'sample_name': sample_name,
                'success': False,
                'error': str(e)
            }
    
    def _compare_fitting_methods(self, img: np.ndarray, edges: np.ndarray,
                                 contours_dict_bs: Dict, contours_dict_nurbs: Dict,
                                 contours_dict_optimized_nurbs: Dict,
                                 contours_dict_optimized_bs: Dict,
                                 segments_info_optimized_nurbs: Optional[Dict] = None,
                                 gt_segments: Optional[List[Dict]] = None) -> Dict:
        """对比不同拟合方法的效果（复用test_curve_fit_metamaterial_antenna.py的逻辑）"""
        # 导入测试类的方法
        from tests.test_curve_fit_metamaterial_antenna import MetamaterialAntennaTester
        
        tester = MetamaterialAntennaTester(output_dir=self.output_dir)
        
        # 设置segments_info（用于优化NURBS）
        if segments_info_optimized_nurbs:
            tester._current_segments_info_optimized_nurbs = segments_info_optimized_nurbs
        
        comparison = tester._compare_fitting_methods(
            img, edges,
            contours_dict_bs, contours_dict_nurbs,
            contours_dict_optimized_nurbs, contours_dict_optimized_bs,
            gt_segments=gt_segments
        )
        
        return comparison
    
    @staticmethod
    def _extract_fitted_points(contour_data: Dict) -> Optional[np.ndarray]:
        """提取拟合点（复用test_curve_fit_metamaterial_antenna.py的逻辑）"""
        from tests.test_curve_fit_metamaterial_antenna import MetamaterialAntennaTester
        return MetamaterialAntennaTester._extract_fitted_points(contour_data)
    
    @staticmethod
    def _draw_polyline_bgr(canvas: np.ndarray, points: np.ndarray, 
                          color: Tuple[int, int, int], thickness: int = 2):
        """绘制折线（复用test_curve_fit_metamaterial_antenna.py的逻辑）"""
        from tests.test_curve_fit_metamaterial_antenna import MetamaterialAntennaTester
        MetamaterialAntennaTester._draw_polyline_bgr(canvas, points, color, thickness)
    
    def _draw_segments_bgr(self, canvas: np.ndarray, segments: List[Dict], 
                           color: Tuple[int, int, int], thickness: int = 2):
        """绘制分段（复用test_curve_fit_metamaterial_antenna.py的逻辑）"""
        from tests.test_curve_fit_metamaterial_antenna import MetamaterialAntennaTester
        tester = MetamaterialAntennaTester()
        tester._draw_segments_bgr(canvas, segments, color, thickness)
    
    def _visualize_method_overview(self,
                                   processed_img: np.ndarray,
                                   edges: np.ndarray,
                                   contours_dict_by_method: Dict[str, Dict],
                                   gt_segments: Optional[List[Dict]],
                                   evaluation_comparison: Optional[Dict],
                                   test_name: str) -> str:
        """
        生成方法对比可视化（使用全称）
        
        参数:
            processed_img: 处理后的图像
            edges: 边缘图像
            contours_dict_by_method: 各方法的轮廓字典
            gt_segments: GT分段
            evaluation_comparison: 评估对比结果
            test_name: 测试名称
        
        返回:
            可视化图像保存路径
        """
        vis_path = os.path.join(self.output_dir, f"{test_name}_methods_overview.png")

        overlays: List[Tuple[str, np.ndarray]] = []

        base = processed_img.copy()
        gt_overlay = base.copy()
        if gt_segments:
            self._draw_segments_bgr(gt_overlay, gt_segments, (0, 0, 255), thickness=2)
        overlays.append(("GT(边界分段)", gt_overlay))

        method_colors = {
            'bspline': (0, 180, 0),
            'nurbs': (180, 0, 0),
            'optimized_nurbs': (180, 0, 180),
            'optimized_bs': (0, 180, 180),
        }

        for method_name, contours_dict in contours_dict_by_method.items():
            overlay = base.copy()
            for contour_data in (contours_dict or {}).values():
                pts = self._extract_fitted_points(contour_data)
                if pts is None:
                    continue
                self._draw_polyline_bgr(overlay, pts, method_colors.get(method_name, (0, 0, 0)), thickness=2)

            # 【关键改进】使用方法全称
            title = self.METHOD_FULL_NAMES.get(method_name, method_name)
            
            # 添加几何感知评分（如果有）
            if evaluation_comparison and method_name in evaluation_comparison:
                ga = evaluation_comparison[method_name].get('geometry_aware', None)
                if isinstance(ga, dict) and 'TotalScore' in ga:
                    title = f"{title}\nGA={ga['TotalScore']:.3f}"
                else:
                    # 如果没有几何感知评分，显示综合得分
                    overall = evaluation_comparison[method_name].get('overall', {}).get('overall', 0)
                    if overall > 0:
                        title = f"{title}\n得分={overall:.2f}"
            
            overlays.append((title, overlay))

        cols = 3
        rows = int(np.ceil((len(overlays) + 2) / cols))
        fig = plt.figure(figsize=(cols * 6, rows * 5))

        ax1 = plt.subplot(rows, cols, 1)
        ax1.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        ax1.set_title("居中图像", fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = plt.subplot(rows, cols, 2)
        ax2.imshow(edges, cmap='gray')
        ax2.set_title("边缘检测", fontsize=12, fontweight='bold')
        ax2.axis('off')

        for i, (title, bgr) in enumerate(overlays, start=3):
            ax = plt.subplot(rows, cols, i)
            ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()

        return vis_path
    
    def run_all_samples(self, max_samples: Optional[int] = None, 
                       complexity_filter: Optional[List[int]] = None,
                       start_index: int = 0) -> Dict:
        """
        运行所有验证样本
        
        参数:
            max_samples: 最大样本数（None表示全部）
            complexity_filter: 复杂度等级过滤（None表示不过滤）
            start_index: 起始索引
        
        返回:
            测试结果字典
        """
        print("\n" + "=" * 70)
        print("开始运行验证集测试")
        print("=" * 70)
        print(f"验证集目录: {self.validation_dataset_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"总样本数: {len(self.summary)}")
        
        # 过滤样本
        samples_to_test = self.summary[start_index:]
        if complexity_filter is not None:
            samples_to_test = [s for s in samples_to_test 
                             if s.get('complexity_level', -1) in complexity_filter]
            print(f"复杂度过滤: {complexity_filter}")
            print(f"过滤后样本数: {len(samples_to_test)}")
        
        if max_samples is not None:
            samples_to_test = samples_to_test[:max_samples]
            print(f"限制样本数: {max_samples}")
        
        print(f"实际测试样本数: {len(samples_to_test)}")
        print("=" * 70)
        
        results = {
            'total': len(samples_to_test),
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        for i, sample in enumerate(samples_to_test):
            result = self.test_single_sample(sample, start_index + i)
            results['tests'].append(result)
            
            if result['success']:
                results['passed'] += 1
            else:
                results['failed'] += 1
        
        # 生成报告
        self._generate_report(results)
        
        return results
    
    def _generate_report(self, results: Dict):
        """生成测试报告"""
        print("\n" + "=" * 70)
        print("测试总结")
        print("=" * 70)
        print(f"总测试数: {results['total']}")
        print(f"通过: {results['passed']}")
        print(f"失败: {results['failed']}")
        if results['total'] > 0:
            print(f"通过率: {results['passed']/results['total']*100:.1f}%")
        
        # 生成对比表格
        comparison_table = self._generate_comparison_table(results)
        if comparison_table:
            print("\n" + "=" * 70)
            print("拟合方法对比表格")
            print("=" * 70)
            if len(comparison_table) > 5000:
                print(comparison_table[:5000] + "\n...(console truncated, full table saved to file)...\n")
            else:
                print(comparison_table)
            
            # 保存对比表格到文件
            table_path = os.path.join(self.output_dir, "comparison_table.txt")
            try:
                with open(table_path, 'w', encoding='utf-8') as f:
                    f.write("验证集拟合方法对比表格\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(comparison_table)
                print(f"\n对比表格已保存: {table_path}")
            except Exception as e:
                print(f"\n[WARN] 对比表格保存失败: {e}")
        
        if self.save_json_report:
            report_path = os.path.join(self.output_dir, "test_report.json")
            try:
                serializable_results = self._convert_to_json_serializable(results)
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, ensure_ascii=False, indent=2)
                print(f"\n报告已保存: {report_path}")
            except Exception as e:
                print(f"\n[WARN] JSON报告保存失败: {e}")
    
    def _convert_to_json_serializable(self, obj, _depth: int = 0, _visited: Optional[set] = None):
        """转换为JSON可序列化对象（复用test_curve_fit_metamaterial_antenna.py的逻辑）"""
        from tests.test_curve_fit_metamaterial_antenna import MetamaterialAntennaTester
        tester = MetamaterialAntennaTester()
        return tester._convert_to_json_serializable(obj, _depth, _visited)
    
    def _generate_comparison_table(self, results: Dict) -> Optional[str]:
        """生成对比表格（复用test_curve_fit_metamaterial_antenna.py的逻辑，但使用方法全称）"""
        from tests.test_curve_fit_metamaterial_antenna import MetamaterialAntennaTester
        
        tester = MetamaterialAntennaTester(output_dir=self.output_dir)
        table = tester._generate_comparison_table(results)
        
        # 替换方法名称为全称
        if table:
            for short_name, full_name in self.METHOD_FULL_NAMES.items():
                table = table.replace(short_name, full_name)
        
        return table


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='验证集测试脚本')
    parser.add_argument('--dataset', type=str, default='validation_dataset',
                       help='验证集目录（默认：validation_dataset）')
    parser.add_argument('--output', type=str, default='validation_test_output',
                       help='输出目录（默认：validation_test_output）')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大测试样本数（默认：全部）')
    parser.add_argument('--complexity', type=int, nargs='+', default=None,
                       help='复杂度等级过滤（例如：--complexity 0 1 2）')
    parser.add_argument('--start-index', type=int, default=0,
                       help='起始索引（默认：0）')
    parser.add_argument('--no-json', action='store_true',
                       help='不保存JSON报告')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ValidationDatasetTester(
        validation_dataset_dir=args.dataset,
        output_dir=args.output,
        save_json_report=not args.no_json
    )
    
    # 运行测试
    results = tester.run_all_samples(
        max_samples=args.max_samples,
        complexity_filter=args.complexity,
        start_index=args.start_index
    )
    
    print("\n所有测试完成!")
    return results


if __name__ == '__main__':
    main()
