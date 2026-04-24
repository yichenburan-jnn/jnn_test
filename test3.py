import pickle
import numpy as np

def verify_constraints(solution_file="matrix_x2.csv"):
    # 1. 加载约束矩阵 G 和边界向量 h
    try:
        with open('G.pkl', 'rb') as f:
            G = pickle.load(f)
        with open('h.pkl', 'rb') as f:
            h = pickle.load(f)
    except FileNotFoundError:
        print("❌ 未找到 G.pkl 或 h.pkl")
        return

    # 2. 加载求解结果 x
    try:
        x = np.loadtxt(solution_file, delimiter=",", skiprows=1) # skiprows=1 跳过表头
    except FileNotFoundError:
        print(f"❌ 未找到求解结果文件 {solution_file}")
        return

    # 3. 数据类型与维度对齐
    G = np.array(G, dtype=np.float64)
    h = np.array(h, dtype=np.float64).flatten()
    x = np.array(x, dtype=np.float64).flatten()

    # 4. 计算 G*x
    # 使用矩阵乘法 @ 计算 Gx
    Gx = G @ x

    # 5. 计算违规量 (Gx - h)
    # 如果 Gx <= h，那么 Gx - h 应该全部小于等于 0
    violation = Gx - h

    # 设定一个科学计算中常用的数值容差
    tolerance = 1e-6

    # 找出所有超出容差的约束索引
    violated_indices = np.where(violation > tolerance)[0]

    # 6. 输出验证结果
    print("--- 约束验证报告 ---")
    print(f"总约束数量: {len(h)}")
    
    # 计算最大的正向违规量
    max_violation = np.max(violation)
    print(f"最大违规量 (Max Gx - h): {max_violation:.4e}")

    if len(violated_indices) == 0:
        print(f"✅ 验证通过！所有 {len(h)} 个约束均满足 (容差 {tolerance})。")
        
        # 补充信息：看看有多少个解是“贴在边界上”的（即 Gx 极其接近 h，起到激活作用的约束）
        active_constraints = np.where(np.abs(violation) < tolerance)[0]
        print(f"💡 激活状态的约束（紧贴边界）数量: {len(active_constraints)} 个")
    else:
        print(f"❌ 验证失败！有 {len(violated_indices)} 个约束未满足。")
        print("未满足的约束索引前 10 个:", violated_indices[:10])
        print("对应的违规数值前 10 个:", violation[violated_indices][:10])

if __name__ == "__main__":
    verify_constraints()
