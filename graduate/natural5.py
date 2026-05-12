import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time
import os

def process_surface_interpolation(input_path, output_path):
    print(f"正在读取输入文件: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"无法找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"无法解析 JSON 文件: {e}")

    # 1. 提取并展平原始观测数据
    x_data = np.array(json_data['x_data']).flatten()
    y_data = np.array(json_data['y_data']).flatten()
    z_data = np.array(json_data['z_data']).flatten()

    # 提取目标网格数据（通常是二维矩阵，例如 401x401）
    xi_data = np.array(json_data['xi'])
    yi_data = np.array(json_data['yi'])

    print(f"观测点总数: {len(x_data)}")
    print(f"目标网格维度: {xi_data.shape} (共 {xi_data.size} 个插值点)")

    # ==========================================
    # 核心步骤 1：重构完美的正交 2D 网格
    # ==========================================
    # 使用 np.round 消除浮点数截断误差（保留7位小数足以满足您的步长精度）
    x_data_rounded = np.round(x_data, decimals=7)
    y_data_rounded = np.round(y_data, decimals=7)

    # 提取唯一坐标轴并排序，确保严格单调递增
    x_axis = np.unique(x_data_rounded)
    y_axis = np.unique(y_data_rounded)
    x_axis.sort()
    y_axis.sort()

    print(f"检测到均匀网格：X轴 {len(x_axis)} 点, Y轴 {len(y_axis)} 点。")

    # 构建 2D Z 矩阵并安全回填数据，彻底免疫原始数据的展开顺序（按行或按列）导致的错位
    Z_grid = np.full((len(x_axis), len(y_axis)), np.nan)
    
    # 寻找每个原始 1D 数据点在二维网格中的行列索引
    x_idx = np.searchsorted(x_axis, x_data_rounded)
    y_idx = np.searchsorted(y_axis, y_data_rounded)
    
    # 将 Z 值填入对应的二维网格位置
    Z_grid[x_idx, y_idx] = z_data

    # ==========================================
    # 核心步骤 2：构建 C 底层极速插值器
    # ==========================================
    print("正在构建基于 PCHIP (分段三次埃尔米特多项式) 的极速插值器...")
    start_time = time.time()
    
    # method='pchip' 能够在保证一阶导数连续（平滑）的同时，严格抑制边缘过冲（无震荡）
    interp = RegularGridInterpolator(
        (x_axis, y_axis), 
        Z_grid, 
        method='pchip', 
        bounds_error=False,   # 允许查询点落在原始数据范围之外（外推）
        fill_value=np.nan     # 外推区域严格填充为 NaN，对齐 MATLAB 行为
    )

    # ==========================================
    # 核心步骤 3：执行批量插值查询
    # ==========================================
    print(f"开始执行 {xi_data.size} 个目标点的插值计算...")
    
    # 将目标的二维网格 xi, yi 组合成 N x 2 的坐标对矩阵
    query_points = np.column_stack((xi_data.flatten(), yi_data.flatten()))
    
    # 将坐标对喂给插值器，瞬间输出结果
    zq_flat = interp(query_points)
    
    # 将展平的插值结果重新塑形为目标网格的维度 (例如 401x401)
    zq_na = zq_flat.reshape(xi_data.shape)
    
    elapsed_time = time.time() - start_time
    print(f"插值计算彻底完成！核心算法耗时: {elapsed_time:.4f} 秒")

    # ==========================================
    # 核心步骤 4：数据清洗与保存
    # ==========================================
    # 将外推产生的 NaN 转换为 JSON 支持的 None (null)
    zq_na_cleaned = np.where(np.isnan(zq_na), None, zq_na).tolist()

    # 更新 JSON 字典
    json_data['zi'] = zq_na_cleaned

    # 写入文件
    print("正在保存文件...")
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 indent=4 保持良好的格式化，方便您用文本编辑器抽查对比
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"成功！新 JSON 文件已保存至: {output_path}")

if __name__ == "__main__":
    # 配置您的实际文件路径
    json_file_path = r'D:\Program\Matlab_natural\griddata_input.json'
    new_json_path  = r'D:\Program\Matlab_natural\natural_py.json'
    
    try:
        process_surface_interpolation(json_file_path, new_json_path)
    except Exception as e:
        print(f"\n运行过程中发生错误:\n{e}")
