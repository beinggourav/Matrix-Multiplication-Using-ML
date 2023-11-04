import numpy as np
# Matrix A - m*n
# Matrix B - n*p
# Normal matrix multiplication - Using three loops
# O(m*n*p) ~ O(n^3)
def naive_multiply(A,B):
    m=len(A)
    n=len(A[0])
    p=len(B[0])
    C=[] # will be of size m*p
    multis=0
    for i in range(m):
        temp=[]
        for j in range(p):
            sum=0
            for k in range(n):
                sum+=A[i][k]*B[k][j]
                multis+=1
            temp.append(sum)
        C.append(temp)
    return multis, np.array(C)