import pickle
import numpy as np
from qpsolvers import solve_qp
def reproduce_and_fix():
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

    res = solve_qb(P,q,G,h,solver="quadprog")
    print(f"quadprog 求解结果:{res}")

if __name__ == "__main__":
    result = reproduce_and_fix()
