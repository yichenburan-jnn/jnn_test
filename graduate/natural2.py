import json
import numpy as np
import os

try:
    from metpy.interpolate import natural_neighbor_to_grid
    HAS_NATURAL = True
except ImportError as e:
    HAS_NATURAL = False
    from scipy.interpolate import griddata as scipy_griddata
    print(f"加载 MetPy 失败: {e}")

def process_surface_interpolation(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"无法找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 1. 提取观测散点数据，必须展平为 1D 数组
    x_data = np.array(json_data['x_data']).flatten()
    y_data = np.array(json_data['y_data']).flatten()
    z_data = np.array(json_data['z_data']).flatten()

    # 2. 提取目标网格点
    xi_data = np.array(json_data['xi'])
    yi_data = np.array(json_data['yi'])

    print(f"清洗前 x_data 观测点数量: {len(x_data)}")

    # ==========================================
    # 数据清洗：合并重复的 (x, y) 坐标点并对 Z 求均值
    # ==========================================
    # 将 x 和 y 合并为 N x 2 的坐标矩阵
    points = np.column_stack((x_data, y_data))

    # 为了防止浮点数极微小误差导致去重失败，先进行合理舍入
    # 您的步长在 0.002~0.003 量级，保留 7 位小数足以保证工业精度
    rounded_points = np.round(points, decimals=7)

    # 寻找唯一的坐标点，并获取索引和出现次数
    unique_points, inverse_indices, counts = np.unique(
        rounded_points, axis=0, return_inverse=True, return_counts=True
    )

    # 累加相同坐标点的 z 值
    z_sum = np.zeros(len(unique_points))
    np.add.at(z_sum, inverse_indices, z_data)

    # 计算平均 z 值
    z_clean = z_sum / counts
    x_clean = unique_points[:, 0]
    y_clean = unique_points[:, 1]

    print(f"清洗后唯一散点数量: {len(x_clean)} (自动移除了 {len(x_data) - len(x_clean)} 个重复坐标点)")
    # ==========================================

    # 3. 执行自然邻近插值
    if HAS_NATURAL:
        print("正在使用 MetPy 计算自然邻近插值 (Sibson)...")
        # 传入清洗后的干净数据
        zq_na = natural_neighbor_to_grid(x_clean, y_clean, z_clean, xi_data, yi_data)
        print(f"插值完成，Zq_na 维度: {zq_na.shape}")
    else:
        print("警告：将降级使用 scipy 的 linear 插值。")
        clean_points = np.column_stack((x_clean, y_clean))
        zq_na = scipy_griddata(clean_points, z_clean, (xi_data, yi_data), method='linear')

    # 4. 数据回填与保存
    zq_na_cleaned = np.where(np.isnan(zq_na), None, zq_na).tolist()
    json_data['zi'] = zq_na_cleaned

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"新 JSON 文件已保存至: {output_path}")


if __name__ == "__main__":
    json_file_path = r'D:\Program\Matlab_natural\griddata_input.json'
    new_json_path  = r'D:\Program\Matlab_natural\natural_py.json'
    
    process_surface_interpolation(json_file_path, new_json_path)
