import pickle
import numpy as np
from qpsolvers import solve_qp

# ... [前面加载 P, q, G, h 的代码保持不变] ...
try:
        with open('P.pkl','rb') as f:
            P = pickle.load(f)
        with open('q.pkl','rb') as f:
            P = pickle.load(f)
        with open('G.pkl','rb') as f:
            P = pickle.load(f)
        with open('h.pkl','rb') as f:
            P = pickle.load(f)
    except FileNotFoundError:
        print("未找到pkl文件")
        return

print("--- 开始进行约束松弛诊断 ---")

# 我们从极微小的松弛开始，逐渐加大，看看在哪个量级下问题变得可行
relax_factors = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

for relax in relax_factors:
    # 将边界条件 h 稍微放宽一点点 (如果是正数就加上松弛量，负数也往“松”的方向走)
    # 简单粗暴的做法是直接在 h 上加上一个微小的正数
    h_relaxed = h + relax 
    
    print(f"\n尝试松弛容差 (Relaxation) = {relax}")
    try:
        res = solve_qp(P, q, G, h_relaxed, solver="osqp", polish=True)
        
        if res is not None:
            print(f"✅ 成功！当约束放宽 {relax} 时，找到了可行解。")
            clean_res = np.where(np.abs(res) < 1e-10, 0.0, res)
            print(f"结果前 5 个数据: {clean_res[:5]}")
            
            # 保存这个解
            np.savetxt("matrix_x2_relaxed.csv", res, delimiter=",", fmt='%.18e', header="solution")
            print("💾 结果已保存至 matrix_x2_relaxed.csv")
            break # 找到解就退出循环
        else:
            print("❌ 依然无解 (primal infeasible)")
            
    except Exception as e:
        print(f"运行出错: {e}")
