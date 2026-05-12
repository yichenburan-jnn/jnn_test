import json
import numpy as np
import os

try:
    from metpy.interpolate import natural_neighbor
    HAS_NATURAL = True
except ImportError:
    HAS_NATURAL = False
    from scipy.interpolate import griddata as scipy_griddata
    print("警告：未找到 metpy 库。将降级使用 scipy 的 cubic 插值。")

def process_surface_interpolation(input_path, output_path):
    # 1. 读取 JSON 文件
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"无法找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"无法解析 JSON 文件: {e}")

    # 2. 提取数据并展平
    # 将输入数据展平为 1D 数组，这是 MetPy 接口的要求
    x_data = np.array(json_data['x_data']).flatten()
    y_data = np.array(json_data['y_data']).flatten()
    z_data = np.array(json_data['z_data']).flatten()

    # 提取目标网格点 (通常为 2D 数组)
    xi_data = np.array(json_data['xi'])
    yi_data = np.array(json_data['yi'])

    print(f"x_data 展平后长度: {len(x_data)}")
    print(f"xi_data (目标网格X) 维度: {xi_data.shape}")

    # 3. 执行自然邻近插值 (Natural Neighbor / Sibson)
    if HAS_NATURAL:
        # metpy 的 natural_neighbor 直接接收展平的 x,y,z 以及目标网格坐标 xi, yi
        print("正在使用 MetPy 计算 2D 自然邻近插值 (Sibson)...")
        zq_na = natural_neighbor(x_data, y_data, z_data, xi_data, yi_data)
        print(f"插值完成，Zq_na 维度: {zq_na.shape}")
    else:
        # 降级方案（仅做保底方案）
        points = np.column_stack((x_data, y_data))
        zq_na = scipy_griddata(points, z_data, (xi_data, yi_data), method='cubic')

    # 4. 数据回填与保存
    # 同样需要处理外推区域的 NaN 值，将其转换为 JSON 支持的 null (None)
    zq_na_cleaned = np.where(np.isnan(zq_na), None, zq_na).tolist()

    json_data['zi'] = zq_na_cleaned

    # 将修改后的数据写入新 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"新 JSON 文件已保存至: {output_path}")


if __name__ == "__main__":
    # 请根据您的实际路径进行修改
    json_file_path = r'D:\Program\Matlab_natural\griddata_input.json'
    new_json_path  = r'D:\Program\Matlab_natural\natural_py.json'
    
    process_surface_interpolation(json_file_path, new_json_path)
