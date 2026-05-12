import json
import numpy as np
import os

try:
    # 必须使用专用的自然邻近库以确保相对误差 < 1%
    from naturalneighbor import griddata as nat_griddata
    HAS_NATURAL = True
except ImportError:
    HAS_NATURAL = False
    from scipy.interpolate import griddata as scipy_griddata
    print("警告：未找到 naturalneighbor 库。将降级使用 scipy 的 cubic 插值，可能无法满足 < 1% 的误差要求。")

def process_surface_interpolation(input_path, output_path):
    # 1. 读取 JSON 文件
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"无法找到输入文件: {input_path}")
        
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"无法解析 JSON 文件: {e}")

    # 2. 提取数据
    # 根据 MATLAB 脚本推测，JSON 中应包含散点数据(x, y, z)和目标网格(xi, yi)
    # 提取散点数据
    x_data = np.array(json_data['x_data'])
    y_data = np.array(json_data['y_data'])
    z_data = np.array(json_data['z_data'])

    # 提取目标网格点（即您提到的 135 x 201 的数据网格）
    xi_data = np.array(json_data['xi'])
    yi_data = np.array(json_data['yi'])

    print(f"x_data 维度: {x_data.shape}")
    print(f"y_data 维度: {y_data.shape}")
    print(f"z_data 维度: {z_data.shape}")
    print(f"xi_data (目标网格X) 维度: {xi_data.shape}")

    # 将输入的散点坐标合并为 N x 2 的数组
    points = np.column_stack((x_data.flatten(), y_data.flatten()))
    values = z_data.flatten()

    # 3. 执行自然邻近插值 (Natural Neighbor)
    if HAS_NATURAL:
        # naturalneighbor.griddata 接受目标网格的 1D 坐标轴序列或 2D 坐标点
        # 这里的 grid_ranges 是为了生成等距网格，或者直接传入目标点
        grid_points = np.column_stack((xi_data.flatten(), yi_data.flatten()))
        
        # 执行 Sibson 自然邻近插值，底层几何逻辑与 MATLAB 高度一致
        zq_flat = nat_griddata(points, values, grid_points)
        zq_na = zq_flat.reshape(xi_data.shape)
        print(f"已使用 naturalneighbor 完成插值，Zq_na 维度: {zq_na.shape}")
    else:
        # 降级方案
        zq_na = scipy_griddata(points, values, (xi_data, yi_data), method='cubic')

    # 4. 数据回填与保存
    # 注意：如果目标点位于散点构成的凸包（Convex Hull）之外，插值结果会产生 NaN (Not a Number)
    # 标准 JSON 格式不支持 NaN，需将其转换为 None (JSON 中的 null)，防止反序列化报错
    zq_na_cleaned = np.where(np.isnan(zq_na), None, zq_na).tolist()

    json_data['zi'] = zq_na_cleaned

    # 将修改后的数据写入新 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 indent=4 保证输出文件的可读性，方便后续算法验证
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"新 JSON 文件已保存至: {output_path}")


if __name__ == "__main__":
    # 路径配置
    json_file_path = r'D:\Program\Matlab_natural\griddata_input.json'
    new_json_path  = r'D:\Program\Matlab_natural\natural_py.json'
    
    process_surface_interpolation(json_file_path, new_json_path)
