import pickle
import numpy as np
from qpsolvers import solve_qp

def solve_and_save_to_csv():
    # 1. 修复加载逻辑（防止 P 覆盖其他变量）
    try:
        with open('P.pkl', 'rb') as f:
            P = pickle.load(f)
        with open('q.pkl', 'rb') as f:
            q = pickle.load(f)
        with open('G.pkl', 'rb') as f:
            G = pickle.load(f)
        with open('h.pkl', 'rb') as f:
            h = pickle.load(f)
        print("✅ 数据加载成功")
    except FileNotFoundError:
        print("❌ 未找到 pkl 文件，请确保 P.pkl, q.pkl, G.pkl, h.pkl 在当前目录下。")
        return

    # 2. 数据预处理
    P = np.array(P, dtype=np.float64)
    q = np.array(q, dtype=np.float64).flatten()
    G = np.array(G, dtype=np.float64)
    h = np.array(h, dtype=np.float64).flatten()
    
    # 对称化 P 矩阵以提高数值稳定性
    P = (P + P.T) / 2

    res = None

    # 3. 策略 A: 优先尝试 OSQP (更接近 MATLAB 的内点法鲁棒性)
    print("正在尝试 OSQP 求解器...")
    res = solve_qp(P, q, G, h, solver="osqp")

    # 4. 策略 B: 如果 OSQP 失败，尝试正则化后的 Quadprog
    if res is None:
        print("OSQP 未能求得解，尝试对 Quadprog 进行对角线正则化...")
        try:
            # 加上微小的扰动 epsilon 使矩阵严格正定
            epsilon = 1e-9
            P_reg = P + epsilon * np.eye(P.shape[0])
            res = solve_qp(P_reg, q, G, h, solver="quadprog")
        except Exception as e:
            print(f"Quadprog 运行出错: {e}")

    # 5. 结果处理与保存
    if res is not None:
        print(f"✅ 求解成功！得到 {len(res)} 个数据。")
        
        # 保存为 CSV 文件
        # fmt='%.18e' 保证了科学计数法的高精度保存
        filename = "matrix_x2.csv"
        np.savetxt(filename, res, delimiter=",", fmt='%.18e', header="solution_x")
        
        print(f"💾 结果已保存至: {filename}")
        
        # 屏幕预览前 5 个非零数据的数量级
        nonzero_count = np.sum(np.abs(res) > 1e-20)
        print(f"预览：非零解数量约 {nonzero_count} 个")
    else:
        print("❌ 所有求解器均返回 None。请检查约束条件 Gx <= h 是否存在可行域。")

if __name__ == "__main__":
    solve_and_save_to_csv()
