import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time

def fast_grid_interpolation(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 提取并重构轴数据
    x_data = np.array(json_data['x']).flatten()
    y_data = np.array(json_data['y']).flatten()
    z_data = np.array(json_data['z']).flatten()

    x_axis = np.unique(np.round(x_data, 7))
    y_axis = np.unique(np.round(y_data, 7))
    x_axis.sort()
    y_axis.sort()

    # 构建 2D Z 矩阵
    Z_grid = np.full((len(x_axis), len(y_axis)), np.nan)
    x_idx = np.searchsorted(x_axis, np.round(x_data, 7))
    y_idx = np.searchsorted(y_axis, np.round(y_data, 7))
    Z_grid[x_idx, y_idx] = z_data

    # 获取目标网格
    xi_data = np.array(json_data['xi'])
    yi_data = np.array(json_data['yi'])
    query_points = np.column_stack((xi_data.flatten(), yi_data.flatten()))

    print("正在构建 C 底层插值器...")
    start_time = time.time()
    
    # 核心：使用 pchip 方法，兼顾自然邻近的平滑性与线性的稳定性（不产生震荡）
    interp = RegularGridInterpolator(
        (x_axis, y_axis), Z_grid, 
        method='pchip', 
        bounds_error=False, fill_value=np.nan
    )

    print("开始执行 160801 个目标点的极速插值...")
    zq_flat = interp(query_points)
    zq_na = zq_flat.reshape(xi_data.shape)
    
    print(f"插值完成！总耗时: {time.time() - start_time:.4f} 秒")

    # 回填并保存
    json_data['zi'] = np.where(np.isnan(zq_na), None, zq_na).tolist()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
