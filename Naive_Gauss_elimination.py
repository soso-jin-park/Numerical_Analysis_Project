import numpy as np
import scipy as sp


def Substitute(a, n, b):            # 후방대입
    """
    pseudo code:
        x_n = b_n / a_n,n
        DOFOR i = n-1, 1, -1
            sum = b_i
            DOFOR j = i+1, n
                sum = sum - a_i,j * x_j
            END DO
            x_i = sum / a_i,i
        END DO
    """
    x = np.zeros(n)
    
    x[n-1] = b[n-1] / a[n-1,n-1]
    
    for i in range(n-2, -1, -1):
        sum = b[i]

        for j in range(i+1, n):
            sum -= a[i,j]*x[j]
        
        x[i] = sum / a[i,i]
    
    return x


def Eliminate(a, n, b, tol):        # 전진소거
    """
    pseudo code:
        DOFOR k = 1, n-1
            DOFOR i = k+1, n
                factor = a_i,k / a_k,k
                DOFOR j = k+1 to n
                    a_i,j = a_i,j - factor * a_k,j
                END DO
                b_i = b_i - factor * b_k
            END DO
        END DO
    """
    er = 0
    
    for k in range(n-1):
        if abs(a[k,k]) < tol:
            er = -1
            return a,b,er
        
        for i in range(k+1,n):
            factor = a[i,k] / a[k,k]
            for j in range(k+1,n):
                a[i,j] = a[i,j] - factor*a[k,j]
            b[i] = b[i] - factor*b[k]
    
    return a,b,er


def Gauss(a, b, n, tol):
    """
    pseudo code:
        er = 0
        CALL Eliminate(a, n, b, tol, er)
        IF er ≠ -1 THEN
            CALL Substitute(a, n, b, x)
        END IF
    """
    a = a.astype(float).copy()  # dtype float 변환 + 원본 보호
    b = b.astype(float).copy()  # dtype float 변환 + 원본 보호
        
    er = 0
    
    a,b,er = Eliminate(a,n,b,tol)
    
    if er != -1:
        x = Substitute(a,n,b)
    else:
        print("특이행렬 - 해를 구할 수 없습니다.")
        return None, er
    
    return x, er


# 실행
if __name__ == "__main__":

    A = np.array([
    [2,  1, -1,  3],
    [4, -2,  5,  1],
    [-1, 3,  2, -4],
    [3, -1,  4,  2]
    ],dtype=float)

    b = np.array([5, 16, -2, 10],dtype=float)
    n   = len(b)
    tol = 1e-10

    print("=" * 30)
    print("   Naïve Gauss Elimination")
    print("=" * 30)
    
    x, er = Gauss(A, b, n, tol)
    
    if er == 0:
        print(f"x = {x[0]:.2f}")
        print(f"y = {x[1]:.2f}")
        print(f"z = {x[2]:.2f}")
        print(f"w = {x[3]:.2f}")
        print(f"\n검증 (Ax=b): {np.allclose(A @ x, b)}")
        print(f"잔차: {np.linalg.norm(A @ x - b):.2e}")


    # NumPy / SciPy groundtruth 비교

    print("\n" + "=" * 40)
    print("   NumPy / SciPy groundtruth 비교")
    print("=" * 40)

    x_np = np.linalg.solve(A.astype(float), b.astype(float))
    x_sp = sp.linalg.solve(A.astype(float), b.astype(float))

    print(f"\n[NumPy]  x={x_np[0]:.2f}  y={x_np[1]:.2f}  z={x_np[2]:.2f}  w={x_np[3]:.2f}")
    print(f"[SciPy]  x={x_sp[0]:.2f}  y={x_sp[1]:.2f}  z={x_sp[2]:.2f}  w={x_sp[3]:.2f}")

    if x is not None:
        diff_np = np.abs(x - x_np)
        diff_sp = np.abs(x - x_sp)

        print("\n--- NumPy 차이 ---")
        print(f"x 차이: {diff_np[0]:.2e}")
        print(f"y 차이: {diff_np[1]:.2e}")
        print(f"z 차이: {diff_np[2]:.2e}")
        print(f"w 차이: {diff_np[3]:.2e}")
        print(f"최대 오차: {np.max(diff_np):.2e}")
        print(f"일치 여부 (tol=1e-10): {np.allclose(x, x_np)}")

        print("\n--- SciPy 차이 ---")
        print(f"x 차이: {diff_sp[0]:.2e}")
        print(f"y 차이: {diff_sp[1]:.2e}")
        print(f"z 차이: {diff_sp[2]:.2e}")
        print(f"w 차이: {diff_sp[3]:.2e}")
        print(f"최대 오차: {np.max(diff_sp):.2e}")
        print(f"일치 여부 (tol=1e-10): {np.allclose(x, x_sp)}")