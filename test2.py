import pickle
import numpy as np
from qpsolvers import solve_qp

def reproduce_and_solve():
    # 1. 修复加载逻辑
    try:
        with open('P.pkl', 'rb') as f:
            P = pickle.load(f)
        with open('q.pkl', 'rb') as f:
            q = pickle.load(f)  
        with open('G.pkl', 'rb') as f:
            G = pickle.load(f)  
        with open('h.pkl', 'rb') as f:
            h = pickle.load(f)  
    except FileNotFoundError:
        print("未找到 pkl 文件")
        return None

    # 2. 格式与维度标准化 (非常重要)
    # qpsolvers 严格要求向量是一维的 (N,)，而不能是二维的列向量 (N, 1)
    P = np.array(P, dtype=np.float64)
    q = np.array(q, dtype=np.float64).flatten() 
    G = np.array(G, dtype=np.float64)
    h = np.array(h, dtype=np.float64).flatten()

    # 3. 强制对称化
    # 消除浮点误差导致的微小不对称
    P = (P + P.T) / 2

    # --- 策略 A: 优先使用 osqp ---
    print("尝试使用 osqp 求解器...")
    try:
        res_osqp = solve_qp(P, q, G, h, solver="osqp")
        if res_osqp is not None:
            print("✅ osqp 求解成功!")
            # 过滤掉极小的浮点误差噪声 (例如将 1e-18 变为 0)
            clean_res = np.where(np.abs(res_osqp) < 1e-10, 0.0, res_osqp)
            print(f"结果前 5 个数据: {clean_res[:5]}")
            return clean_res
        else:
            print("❌ osqp 返回了 None，可能存在约束冲突。")
    except Exception as e:
        print(f"osqp 运行时报错: {e}")

    # --- 策略 B: 使用正则化 + quadprog ---
    print("\n尝试使用正则化后的 quadprog 求解器...")
    try:
        # 在对角线上增加 1e-8，强制矩阵严格正定
        epsilon = 1e-8
        P_regularized = P + epsilon * np.eye(P.shape[0])
        
        res_quad = solve_qp(P_regularized, q, G, h, solver="quadprog")
        if res_quad is not None:
            print("✅ quadprog (正则化后) 求解成功!")
            clean_res = np.where(np.abs(res_quad) < 1e-10, 0.0, res_quad)
            print(f"结果前 5 个数据: {clean_res[:5]}")
            return clean_res
        else:
            print("❌ quadprog 依然返回 None。")
    except Exception as e:
        print(f"quadprog 运行时报错: {e}")

    print("\n⚠️ 两种求解器均未能求解。请检查 G 和 h 构成的约束条件是否存在无解的情况。")
    return None

if __name__ == "__main__":
    result = reproduce_and_solve()
