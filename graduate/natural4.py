import json
import numpy as np
import os
from metpy.interpolate import natural_neighbor_to_grid

def process_surface_interpolation(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"无法找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 1. 提取数据
    x_data = np.array(json_data['x_data']).flatten()
    y_data = np.array(json_data['y_data']).flatten()
    z_data = np.array(json_data['z_data']).flatten()

    xi_data = np.array(json_data['xi'])
    yi_data = np.array(json_data['yi'])

    # ==========================================
    # 核心黑科技：引入极微小扰动 (Joggle)
    # 打破完美矩形网格导致的 Voronoi 拓扑奇异性（共圆退化）
    # 1e-10 的噪声远低于您的测量精度，但能拯救几何算法
    # ==========================================
    np.random.seed(42) # 固定随机种子，保证每次运行结果完全一致
    x_joggled = x_data + np.random.normal(0, 1e-10, size=x_data.shape)
    y_joggled = y_data + np.random.normal(0, 1e-10, size=y_data.shape)

    print("已添加微小扰动，正在使用 MetPy 计算 2D 自然邻近插值 (Sibson)...")
    
    # 使用扰动后的坐标和原始 Z 值进行精确的 Sibson 插值
    zq_na = natural_neighbor_to_grid(x_joggled, y_joggled, z_data, xi_data, yi_data)
    
    print(f"插值完成，Zq_na 维度: {zq_na.shape}")

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
