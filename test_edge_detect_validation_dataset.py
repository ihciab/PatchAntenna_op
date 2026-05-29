"""
边缘检测验证集测试脚本
专门用于测试和对比多种边缘检测改进方案

功能：
1. 加载validation_dataset中的图像
2. 应用多种边缘检测方案
3. 对比评估各方案效果
4. 生成对比报告和可视化
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
from core.image.initializer import ImageInitializer
# 已删除：多通道边缘检测和自动选择策略效果较差，不再使用
# from core.image.improved_edge_detector import ImprovedEdgeDetector, EdgeDetectionStrategy, create_improved_edge_detector
from core.image.gt_edge_generator import GTEdgeGenerator, generate_gt_edges_from_json_file
from core.image.geometry_aware_edge_extractor import GeometryAwareEdgeExtractor, create_geometry_aware_extractor

plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei"]


class EdgeDetectionTester:
    """边缘检测测试类"""
    
    # 边缘检测方法全称映射
    METHOD_FULL_NAMES = {
        'original': '原始边缘检测（固定阈值）',
        'geometry_aware': '几何感知边缘提取（固定阈值+结构筛选）',
    }
    
    def __init__(self, validation_dataset_dir: str = "validation_dataset",
                 output_dir: str = "validation_test_edge_detect",
                 save_json_report: bool = True):
        """
        初始化边缘检测测试器
        
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
    
    def _original_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        原始边缘检测方法（固定阈值）
        
        参数:
            image: 输入图像（BGR格式）
        
        返回:
            边缘图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 固定阈值Canny（原始方法）
        edges = cv2.Canny(blurred, 2500, 5000, apertureSize=5)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        return edges
    
    def _evaluate_edges(self, edges: np.ndarray, original_image: np.ndarray, 
                       gt_edges: Optional[np.ndarray] = None) -> Dict:
        """
        评估边缘检测结果（改进版：支持与GT对比）
        
        参数:
            edges: 边缘图像
            original_image: 原始图像
            gt_edges: 真值边缘图像（可选）
        
        返回:
            评估结果字典
        """
        # 基本统计
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # 检测轮廓
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_count = len(contours)
        
        # 计算轮廓总长度
        total_contour_length = sum(cv2.arcLength(c, True) for c in contours)
        
        # 计算边缘连续性（通过轮廓数量与总长度的比值）
        avg_contour_length = total_contour_length / contour_count if contour_count > 0 else 0.0
        
        # 计算边缘密度（边缘像素密度）
        edge_density = edge_pixels / (original_image.shape[0] * original_image.shape[1]) if original_image.size > 0 else 0.0
        
        # 计算边缘连通性（通过轮廓数量评估）
        # 连通性好的边缘，轮廓数量应该适中（不能太多也不能太少）
        connectivity_score = 1.0 / (1.0 + abs(contour_count - 10) / 10.0)  # 假设理想轮廓数为10
        
        result = {
            'edge_pixel_count': int(edge_pixels),
            'edge_ratio': float(edge_ratio),
            'edge_density': float(edge_density),
            'contour_count': int(contour_count),
            'total_contour_length': float(total_contour_length),
            'avg_contour_length': float(avg_contour_length),
            'connectivity_score': float(connectivity_score),
        }
        
        # 如果有GT，计算与GT的对比指标
        if gt_edges is not None:
            gt_metrics = self._compare_with_gt(edges, gt_edges)
            result.update(gt_metrics)
        
        return result
    
    def _compare_with_gt(self, detected_edges: np.ndarray, gt_edges: np.ndarray, 
                         tolerance_pixels: int = 2) -> Dict:
        """
        与真值边缘对比（改进版：使用距离变换容忍小的位置偏差）
        
        参数:
            detected_edges: 检测到的边缘
            gt_edges: 真值边缘
            tolerance_pixels: 位置偏差容忍度（像素），默认2像素
        
        返回:
            对比指标字典
        """
        # 确保尺寸一致
        if detected_edges.shape != gt_edges.shape:
            h, w = min(detected_edges.shape[0], gt_edges.shape[0]), min(detected_edges.shape[1], gt_edges.shape[1])
            detected_edges = detected_edges[:h, :w]
            gt_edges = gt_edges[:h, :w]
        
        # 方法1：严格像素匹配（原始方法）
        intersection_strict = np.logical_and(detected_edges > 0, gt_edges > 0).sum()
        union_strict = np.logical_or(detected_edges > 0, gt_edges > 0).sum()
        iou_strict = intersection_strict / union_strict if union_strict > 0 else 0.0
        
        detected_pixels = np.count_nonzero(detected_edges)
        gt_pixels = np.count_nonzero(gt_edges)
        precision_strict = intersection_strict / detected_pixels if detected_pixels > 0 else 0.0
        recall_strict = intersection_strict / gt_pixels if gt_pixels > 0 else 0.0
        
        # 方法2：容忍位置偏差的匹配（使用距离变换）
        # 对GT边缘进行膨胀，容忍tolerance_pixels像素的位置偏差
        if tolerance_pixels > 0 and gt_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (2 * tolerance_pixels + 1, 2 * tolerance_pixels + 1))
            gt_dilated = cv2.dilate(gt_edges, kernel, iterations=1)
            
            # 计算容忍偏差后的匹配
            intersection_tolerant = np.logical_and(detected_edges > 0, gt_dilated > 0).sum()
            union_tolerant = np.logical_or(detected_edges > 0, gt_edges > 0).sum()
            iou_tolerant = intersection_tolerant / union_tolerant if union_tolerant > 0 else 0.0
            
            precision_tolerant = intersection_tolerant / detected_pixels if detected_pixels > 0 else 0.0
            recall_tolerant = intersection_tolerant / gt_pixels if gt_pixels > 0 else 0.0
        else:
            # 如果tolerance_pixels为0或GT为空，使用严格匹配
            iou_tolerant = iou_strict
            precision_tolerant = precision_strict
            recall_tolerant = recall_strict
        
        # F1 Score（使用容忍偏差的匹配）
        f1_score = 2 * (precision_tolerant * recall_tolerant) / (precision_tolerant + recall_tolerant) if (precision_tolerant + recall_tolerant) > 0 else 0.0
        
        # 假阳性率（False Positive Rate）
        false_positives = np.logical_and(detected_edges > 0, gt_edges == 0).sum()
        fpr = false_positives / (detected_edges.size - gt_pixels) if (detected_edges.size - gt_pixels) > 0 else 0.0
        
        # 假阴性率（False Negative Rate）
        false_negatives = np.logical_and(detected_edges == 0, gt_edges > 0).sum()
        fnr = false_negatives / gt_pixels if gt_pixels > 0 else 0.0
        
        # 轮廓匹配度（通过轮廓数量对比）
        detected_contours, _ = cv2.findContours(detected_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        gt_contours, _ = cv2.findContours(gt_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_match_ratio = len(gt_contours) / len(detected_contours) if len(detected_contours) > 0 else 0.0
        contour_count_error = abs(len(detected_contours) - len(gt_contours)) / max(len(gt_contours), 1)
        
        return {
            # 严格匹配指标（用于参考）
            'iou_strict': float(iou_strict),
            'precision_strict': float(precision_strict),
            'recall_strict': float(recall_strict),
            # 容忍偏差的匹配指标（主要指标）
            'iou': float(iou_tolerant),  # 主要IoU指标
            'precision': float(precision_tolerant),  # 主要Precision指标
            'recall': float(recall_tolerant),  # 主要Recall指标
            'f1_score': float(f1_score),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'gt_contour_count': int(len(gt_contours)),
            'contour_match_ratio': float(contour_match_ratio),
            'contour_count_error': float(contour_count_error),
            # 调试信息
            'detected_pixels': int(detected_pixels),
            'gt_pixels': int(gt_pixels),
            'tolerance_pixels': int(tolerance_pixels),
        }
    
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
        
        # 加载图像
        image_path = sample.get('image_path', '')
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.validation_dataset_dir, 
                                     image_path.replace('validation_dataset\\', '').replace('validation_dataset/', ''))
        
        if not os.path.exists(image_path):
            # 尝试从images目录加载
            images_dir = os.path.join(self.validation_dataset_dir, "images")
            image_path = os.path.join(images_dir, f"{sample_name}.png")
        
        if not os.path.exists(image_path):
            print(f"   [ERROR] 图像文件不存在: {image_path}")
            return {
                'sample_name': sample_name,
                'success': False,
                'error': f'图像文件不存在: {image_path}'
            }
        
        try:
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            print(f"   [OK] 图像尺寸: {image.shape}")
            
            # 加载GT并生成真值边缘
            gt_edges = None
            gt_path = sample.get('gt_path', '')
            if not os.path.isabs(gt_path):
                gt_path = os.path.join(self.validation_dataset_dir, 
                                      gt_path.replace('validation_dataset\\', '').replace('validation_dataset/', ''))
            
            if not os.path.exists(gt_path):
                # 尝试从gt目录加载
                gt_dir = os.path.join(self.validation_dataset_dir, "gt")
                gt_path = os.path.join(gt_dir, f"{sample_name}.json")
            
            if os.path.exists(gt_path):
                try:
                    print(f"\n0. 加载GT并生成真值边缘...")
                    # 加载GT JSON以获取几何信息
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        gt_json = json.load(f)
                    geometry = gt_json.get('geometry', {})
                    geometry_type = geometry.get('type', 'unknown')
                    center = geometry.get('center', None)
                    
                    print(f"   [INFO] 几何类型: {geometry_type}")
                    if center:
                        print(f"   [INFO] 中心点: {center}")
                    else:
                        print(f"   [INFO] 中心点: 默认图像中心")
                    
                    # 生成GT边缘
                    gt_edges = generate_gt_edges_from_json_file(gt_path, img_size=image.shape[0])
                    gt_contours, _ = cv2.findContours(gt_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    gt_pixel_count = np.count_nonzero(gt_edges)
                    
                    print(f"   [OK] GT轮廓数: {len(gt_contours)}, GT边缘像素数: {gt_pixel_count}")
                    
                    # 检查GT边缘是否为空（某些类型无法重建）
                    if gt_pixel_count == 0:
                        print(f"   [WARN] GT边缘为空！几何类型 '{geometry_type}' 可能无法从参数精确重建")
                        print(f"   [WARN] 评估结果可能不准确（F1/IoU将为0）")
                except Exception as e:
                    import traceback
                    print(f"   [WARN] 无法生成GT边缘: {e}")
                    print(f"   [WARN] 详细错误: {traceback.format_exc()}")
                    gt_edges = None
            else:
                print(f"   [WARN] GT文件不存在: {gt_path}")
            
            # 应用多种边缘检测方法
            edge_results = {}
            evaluation_results = {}
            
            # 1. 原始方法（固定阈值）
            print("\n1. 原始边缘检测（固定阈值）...")
            start_time = time.time()
            edges_original = self._original_edge_detection(image)
            time_original = time.time() - start_time
            eval_original = self._evaluate_edges(edges_original, image, gt_edges)
            edge_results['original'] = edges_original
            evaluation_results['original'] = {
                **eval_original,
                'time': time_original
            }
            print(f"   [OK] 轮廓数: {eval_original['contour_count']}, "
                  f"边缘密度: {eval_original['edge_density']:.4f}, "
                  f"耗时: {time_original:.3f}s", end='')
            if gt_edges is not None and 'f1_score' in eval_original:
                print(f", F1: {eval_original['f1_score']:.4f}, "
                      f"IoU: {eval_original['iou']:.4f}")
            else:
                print()
            
            # 2. 几何感知边缘提取（固定阈值 + 结构筛选）
            print("\n2. 几何感知边缘提取（固定阈值+结构筛选）...")
            start_time = time.time()
            extractor = create_geometry_aware_extractor(
                min_contour_length_ratio=0.005,  # 最小轮廓长度为图像周长的0.5%
                max_curvature_variance=0.5,      # 最大曲率方差（更严格）
                line_fit_threshold=2.0,          # 直线拟合误差阈值（像素）
                arc_fit_threshold=2.0,           # 圆弧拟合误差阈值（像素）
                min_arc_radius=5.0,              # 最小圆弧半径
                max_arc_radius=500.0,           # 最大圆弧半径
                prefer_closed=True,              # 优先闭合轮廓
                min_points_for_fit=5,            # 拟合所需的最小点数
                geometry_score_threshold=0.5,    # 几何评分阈值（0-1）
                use_gt_prior=True                # 使用GT先验规则
            )
            # 使用与原始方法相同的Canny参数
            edges_geometry_aware = extractor.extract_edges(image, canny_low=2500, canny_high=5000)
            time_geometry_aware = time.time() - start_time
            eval_geometry_aware = self._evaluate_edges(edges_geometry_aware, image, gt_edges)
            edge_results['geometry_aware'] = edges_geometry_aware
            evaluation_results['geometry_aware'] = {
                **eval_geometry_aware,
                'time': time_geometry_aware
            }
            print(f"   [OK] 轮廓数: {eval_geometry_aware['contour_count']}, "
                  f"边缘密度: {eval_geometry_aware['edge_density']:.4f}, "
                  f"耗时: {time_geometry_aware:.3f}s", end='')
            if gt_edges is not None and 'f1_score' in eval_geometry_aware:
                print(f", F1: {eval_geometry_aware['f1_score']:.4f}, "
                      f"IoU: {eval_geometry_aware['iou']:.4f}")
            else:
                print()
            
            # 生成可视化
            vis_path = self._visualize_comparison(
                image, edge_results, evaluation_results, sample_name, gt_edges
            )
            
            # 生成对比表格
            table_path = self._generate_comparison_table(
                evaluation_results, sample_name
            )
            
            result = {
                'sample_name': sample_name,
                'success': True,
                'image_path': image_path,
                'edge_results': {k: {
                    'edge_pixel_count': v['edge_pixel_count'],
                    'contour_count': v['contour_count'],
                    'edge_density': v['edge_density'],
                    'time': v['time']
                } for k, v in evaluation_results.items()},
                'evaluation': evaluation_results,
                'visualization_path': vis_path,
                'comparison_table_path': table_path,
                'complexity_level': sample.get('complexity_level', -1),
                'structure_tags': sample.get('structure_tags', [])
            }
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            print(f"   [ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'sample_name': sample_name,
                'success': False,
                'error': str(e)
            }
    
    def _visualize_comparison(self, 
                             original_image: np.ndarray,
                             edge_results: Dict[str, np.ndarray],
                             evaluation_results: Dict[str, Dict],
                             sample_name: str,
                             gt_edges: Optional[np.ndarray] = None) -> str:
        """
        可视化对比结果
        
        参数:
            original_image: 原始图像
            edge_results: 各方法的边缘检测结果
            evaluation_results: 各方法的评估结果
            sample_name: 样本名称
        
        返回:
            可视化图像保存路径
        """
        num_methods = len(edge_results)
        cols = 3
        # 如果有GT，增加一行显示GT
        has_gt = gt_edges is not None
        rows = (num_methods + 2 + (1 if has_gt else 0) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        ax_idx = 0
        
        # 显示原始图像
        axes[ax_idx].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[ax_idx].set_title('原始图像', fontsize=12, fontweight='bold')
        axes[ax_idx].axis('off')
        ax_idx += 1
        
        # 显示GT边缘（如果有）
        if has_gt:
            overlay_gt = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB).copy()
            gt_colored = np.zeros_like(overlay_gt)
            gt_colored[:, :, 1] = gt_edges  # 绿色GT边缘
            overlay_gt = cv2.addWeighted(overlay_gt, 0.7, gt_colored, 0.3, 0)
            axes[ax_idx].imshow(overlay_gt)
            gt_contours, _ = cv2.findContours(gt_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            axes[ax_idx].set_title(f'真值边缘（GT）\n轮廓数: {len(gt_contours)}', 
                                  fontsize=12, fontweight='bold', color='green')
            axes[ax_idx].axis('off')
            ax_idx += 1
        
        # 显示各方法的边缘检测结果
        method_order = ['original', 'geometry_aware']
        
        for idx, method in enumerate(method_order):
            if method not in edge_results:
                continue
            
            if ax_idx >= len(axes):
                break
            
            edges = edge_results[method]
            eval_info = evaluation_results[method]
            method_name = self.METHOD_FULL_NAMES.get(method, method)
            
            # 叠加显示（原始图像 + 边缘）
            overlay = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB).copy()
            edges_colored = np.zeros_like(overlay)
            edges_colored[:, :, 0] = edges  # 红色边缘
            
            # 如果有GT，同时显示GT（绿色）
            if has_gt:
                gt_colored = np.zeros_like(overlay)
                gt_colored[:, :, 1] = gt_edges  # 绿色GT边缘
                overlay = cv2.addWeighted(overlay, 0.6, edges_colored, 0.2, 0)
                overlay = cv2.addWeighted(overlay, 1.0, gt_colored, 0.2, 0)
            else:
                overlay = cv2.addWeighted(overlay, 0.7, edges_colored, 0.3, 0)
            
            axes[ax_idx].imshow(overlay)
            title = f"{method_name}\n"
            title += f"轮廓数: {eval_info['contour_count']}, "
            title += f"密度: {eval_info['edge_density']:.4f}\n"
            title += f"耗时: {eval_info['time']:.3f}s"
            if has_gt and 'f1_score' in eval_info:
                title += f"\nF1: {eval_info['f1_score']:.3f}, IoU: {eval_info['iou']:.3f}"
            axes[ax_idx].set_title(title, fontsize=10)
            axes[ax_idx].axis('off')
            ax_idx += 1
        
        # 隐藏多余的子图
        for idx in range(ax_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        vis_path = os.path.join(self.output_dir, f"{sample_name}_edge_comparison.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   [OK] 可视化已保存: {vis_path}")
        return vis_path
    
    def _generate_comparison_table(self, 
                                  evaluation_results: Dict[str, Dict],
                                  sample_name: str) -> str:
        """
        生成对比表格
        
        参数:
            evaluation_results: 各方法的评估结果
            sample_name: 样本名称
        
        返回:
            表格文件保存路径
        """
        table_lines = []
        table_lines.append("=" * 80)
        table_lines.append(f"边缘检测方法对比 - {sample_name}")
        table_lines.append("=" * 80)
        table_lines.append("")
        if any('f1_score' in v for v in evaluation_results.values()):
            table_lines.append(f"{'方法':<25} {'轮廓数':<10} {'边缘密度':<12} {'F1分数':<12} {'IoU':<12} {'耗时(s)':<10}")
        else:
            table_lines.append(f"{'方法':<25} {'轮廓数':<10} {'边缘密度':<12} {'连通性':<12} {'耗时(s)':<10}")
        table_lines.append("-" * 80)
        
        method_order = ['original', 'geometry_aware']
        
        for method in method_order:
            if method not in evaluation_results:
                continue
            
            eval_info = evaluation_results[method]
            method_name = self.METHOD_FULL_NAMES.get(method, method)
            
            if 'f1_score' in eval_info:
                table_lines.append(
                    f"{method_name:<25} "
                    f"{eval_info['contour_count']:<10} "
                    f"{eval_info['edge_density']:<12.6f} "
                    f"{eval_info['f1_score']:<12.4f} "
                    f"{eval_info['iou']:<12.4f} "
                    f"{eval_info['time']:<10.3f}"
                )
            else:
                table_lines.append(
                    f"{method_name:<25} "
                    f"{eval_info['contour_count']:<10} "
                    f"{eval_info['edge_density']:<12.6f} "
                    f"{eval_info['connectivity_score']:<12.4f} "
                    f"{eval_info['time']:<10.3f}"
                )
        
        table_lines.append("")
        table_lines.append("=" * 80)
        
        table_text = "\n".join(table_lines)
        
        # 保存表格
        table_path = os.path.join(self.output_dir, f"{sample_name}_edge_comparison_table.txt")
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(table_text)
        
        print(f"   [OK] 对比表格已保存: {table_path}")
        return table_path
    
    def run_test_suite(self, max_samples: Optional[int] = None,
                      complexity_filter: Optional[List[int]] = None) -> Dict:
        """
        运行测试套件
        
        参数:
            max_samples: 最大测试样本数（None表示全部）
            complexity_filter: 复杂度等级过滤（None表示全部）
        
        返回:
            测试报告字典
        """
        print("\n" + "=" * 70)
        print("开始边缘检测测试套件")
        print("=" * 70)
        
        # 过滤样本
        test_samples = self.summary.copy()
        
        if complexity_filter is not None:
            test_samples = [s for s in test_samples 
                          if s.get('complexity_level', -1) in complexity_filter]
            print(f"复杂度过滤: {complexity_filter}, 剩余 {len(test_samples)} 个样本")
        
        if max_samples is not None:
            test_samples = test_samples[:max_samples]
            print(f"限制样本数: {max_samples}")
        
        print(f"将测试 {len(test_samples)} 个样本")
        print("=" * 70)
        
        # 运行测试
        for idx, sample in enumerate(test_samples):
            self.test_single_sample(sample, idx)
        
        # 生成总结报告
        summary_report = self._generate_summary_report()
        
        # 保存JSON报告
        if self.save_json_report:
            report_path = os.path.join(self.output_dir, "test_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': summary_report,
                    'detailed_results': self.test_results
                }, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] JSON报告已保存: {report_path}")
        
        return summary_report
    
    def _generate_summary_report(self) -> Dict:
        """
        生成总结报告
        
        返回:
            总结报告字典
        """
        if not self.test_results:
            return {}
        
        # 统计各方法的平均指标
        method_stats = {}
        method_order = ['original', 'geometry_aware']
        
        for method in method_order:
            method_stats[method] = {
                'contour_count': [],
                'edge_density': [],
                'connectivity_score': [],
                'time': []
            }
        
        for result in self.test_results:
            if not result.get('success', False):
                continue
            
            for method, eval_info in result.get('evaluation', {}).items():
                if method in method_stats:
                    method_stats[method]['contour_count'].append(eval_info['contour_count'])
                    method_stats[method]['edge_density'].append(eval_info['edge_density'])
                    method_stats[method]['connectivity_score'].append(eval_info['connectivity_score'])
                    method_stats[method]['time'].append(eval_info['time'])
        
        # 计算平均值
        summary = {}
        for method, stats in method_stats.items():
            if stats['contour_count']:
                summary[method] = {
                    'avg_contour_count': float(np.mean(stats['contour_count'])),
                    'avg_edge_density': float(np.mean(stats['edge_density'])),
                    'avg_connectivity_score': float(np.mean(stats['connectivity_score'])),
                    'avg_time': float(np.mean(stats['time'])),
                    'method_name': self.METHOD_FULL_NAMES.get(method, method)
                }
        
        # 如果有GT指标，也计算平均值
        gt_method_stats = {}
        for method in method_order:
            gt_method_stats[method] = {
                'f1_score': [],
                'iou': [],
                'precision': [],
                'recall': []
            }
        
        for result in self.test_results:
            if not result.get('success', False):
                continue
            
            for method, eval_info in result.get('evaluation', {}).items():
                if method in gt_method_stats:
                    if 'f1_score' in eval_info:
                        gt_method_stats[method]['f1_score'].append(eval_info['f1_score'])
                        gt_method_stats[method]['iou'].append(eval_info['iou'])
                        gt_method_stats[method]['precision'].append(eval_info['precision'])
                        gt_method_stats[method]['recall'].append(eval_info['recall'])
        
        # 添加到summary
        for method, stats in gt_method_stats.items():
            if method in summary and stats['f1_score']:
                summary[method]['avg_f1_score'] = float(np.mean(stats['f1_score']))
                summary[method]['avg_iou'] = float(np.mean(stats['iou']))
                summary[method]['avg_precision'] = float(np.mean(stats['precision']))
                summary[method]['avg_recall'] = float(np.mean(stats['recall']))
        
        # 打印总结
        print("\n" + "=" * 70)
        print("测试总结报告")
        print("=" * 70)
        
        # 检查是否有GT指标
        has_gt_metrics = any('avg_f1_score' in s for s in summary.values())
        
        if has_gt_metrics:
            print(f"\n{'方法':<25} {'平均轮廓数':<15} {'平均F1分数':<15} {'平均IoU':<15} {'平均耗时(s)':<15}")
            print("-" * 70)
            
            for method in method_order:
                if method in summary:
                    s = summary[method]
                    print(f"{s['method_name']:<25} "
                          f"{s['avg_contour_count']:<15.2f} "
                          f"{s.get('avg_f1_score', 0.0):<15.4f} "
                          f"{s.get('avg_iou', 0.0):<15.4f} "
                          f"{s['avg_time']:<15.3f}")
        else:
            print(f"\n{'方法':<25} {'平均轮廓数':<15} {'平均边缘密度':<15} {'平均连通性':<15} {'平均耗时(s)':<15}")
            print("-" * 70)
            
            for method in method_order:
                if method in summary:
                    s = summary[method]
                    print(f"{s['method_name']:<25} "
                          f"{s['avg_contour_count']:<15.2f} "
                          f"{s['avg_edge_density']:<15.6f} "
                          f"{s['avg_connectivity_score']:<15.4f} "
                          f"{s['avg_time']:<15.3f}")
        
        print("=" * 70)
        
        return {
            'total_samples': len(self.test_results),
            'successful_samples': sum(1 for r in self.test_results if r.get('success', False)),
            'method_statistics': summary
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='边缘检测验证集测试')
    parser.add_argument('--validation-dataset', type=str, default='validation_dataset',
                       help='验证集目录路径')
    parser.add_argument('--output', type=str, default='validation_test_edge_detect',
                       help='输出目录')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大测试样本数（默认：全部）')
    parser.add_argument('--complexity', type=int, nargs='+', default=None,
                       help='复杂度等级过滤（例如：--complexity 0 1 2）')
    parser.add_argument('--no-json', action='store_true',
                       help='不保存JSON报告')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = EdgeDetectionTester(
        validation_dataset_dir=args.validation_dataset,
        output_dir=args.output,
        save_json_report=not args.no_json
    )
    
    # 运行测试
    tester.run_test_suite(
        max_samples=args.max_samples,
        complexity_filter=args.complexity
    )
    
    print("\n测试完成！")


if __name__ == '__main__':
    main()
