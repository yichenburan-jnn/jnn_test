import json
import numpy as np
import os

try:
    # 针对 MetPy 1.0+ (如 1.7.1) 的正确导入方式
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

    # 1. 提取观测散点数据，必须展平为 1D 数组 (MetPy 的输入要求)
    x_data = np.array(json_data['x_data']).flatten()
    y_data = np.array(json_data['y_data']).flatten()
    z_data = np.array(json_data['z_data']).flatten()

    # 2. 提取目标网格点，保留为 2D 数组 (M, N 维度)
    xi_data = np.array(json_data['xi'])
    yi_data = np.array(json_data['yi'])

    print(f"x_data 观测点数量: {len(x_data)}")
    print(f"xi_data (目标网格) 维度: {xi_data.shape}")

    # 3. 执行自然邻近插值
    if HAS_NATURAL:
        print("正在使用 MetPy 计算自然邻近插值 (Sibson)...")
        # 传入：观测点的 x、y、z (1D) 以及 目标网格的 X、Y (2D)
        zq_na = natural_neighbor_to_grid(x_data, y_data, z_data, xi_data, yi_data)
        print(f"插值完成，Zq_na 维度: {zq_na.shape}")
    else:
        print("警告：将降级使用 scipy 的 linear 插值。")
        points = np.column_stack((x_data, y_data))
        zq_na = scipy_griddata(points, z_data, (xi_data, yi_data), method='linear')

    # 4. 数据回填与保存 (外推区域填充为 None)
    zq_na_cleaned = np.where(np.isnan(zq_na), None, zq_na).tolist()
    json_data['zi'] = zq_na_cleaned

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"新 JSON 文件已保存至: {output_path}")


if __name__ == "__main__":
    json_file_path = r'D:\Program\Matlab_natural\griddata_input.json'
    new_json_path  = r'D:\Program\Matlab_natural\natural_py.json'
    
    process_surface_interpolation(json_file_path, new_json_path)
