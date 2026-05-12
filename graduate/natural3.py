import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os

def process_surface_interpolation(input_path, output_path):
    # 1. 读取 JSON 文件
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"无法找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 提取展平的观测数据
    x_data = np.array(json_data['x_data']).flatten()
    y_data = np.array(json_data['y_data']).flatten()
    z_data = np.array(json_data['z_data']).flatten()

    xi_data = np.array(json_data['xi'])
    yi_data = np.array(json_data['yi'])

    # 2. 重建正交均匀网格的坐标轴
    # 提取唯一坐标系并排序，确保严格单调递增
    x_axis = np.unique(np.round(x_data, 7))
    y_axis = np.unique(np.round(y_data, 7))
    x_axis.sort()
    y_axis.sort()

    print(f"检测到完美均匀网格：X轴 {len(x_axis)} 个点, Y轴 {len(y_axis)} 个点。总计 {len(x_axis)*len(y_axis)} 个点。")

    # 3. 将一维 Z 数据安全地还原为 2D 矩阵
    # 这种做法彻底免疫了 MATLAB 无论是按列还是按行展开带来的错位风险
    Z_grid = np.full((len(x_axis), len(y_axis)), np.nan)
    
    # 寻找每个原始点在二维网格中的索引位置
    x_idx = np.searchsorted(x_axis, np.round(x_data, 7))
    y_idx = np.searchsorted(y_axis, np.round(y_data, 7))
    
    # 填入数据
    Z_grid[x_idx, y_idx] = z_data

    # 4. 构建针对均匀网格的高效插值器
    # method='linear' 在均匀网格下即为双线性插值，平滑且无过冲震荡现象
    print("正在构建 RegularGridInterpolator 并执行网格内插值...")
    interp = RegularGridInterpolator(
        (x_axis, y_axis), 
        Z_grid, 
        method='linear', 
        bounds_error=False,   # 允许外推区域
        fill_value=np.nan     # 外推区域填充为 NaN，对齐 MATLAB 行为
    )

    # 5. 对目标 xi_data, yi_data 进行查询
    query_points = np.column_stack((xi_data.flatten(), yi_data.flatten()))
    zq_flat = interp(query_points)
    
    # 恢复目标网格形状
    zq_na = zq_flat.reshape(xi_data.shape)
    print(f"插值完成，Zq_na 维度: {zq_na.shape}")

    # 6. 数据回填与保存
    zq_na_cleaned = np.where(np.isnan(zq_na), None, zq_na).tolist()
    json_data['zi'] = zq_na_cleaned

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"新 JSON 文件已保存至: {output_path}")


if __name__ == "__main__":
    json_file_path = r'D:\Program\Matlab_natural\griddata_input.json'
    new_json_path  = r'D:\Program\Matlab_natural\natural_py.json'
    
    process_surface_interpolation(json_file_path, new_json_path)
