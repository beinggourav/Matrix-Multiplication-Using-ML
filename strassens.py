import numpy as np
from naive import naive_multiply

def combine_matrix(C11,C12,C21,C22):
    n=len(C11)
    result=[[0 for j in range(2*n)] for i in range(2*n)]
    for i in range(n):
        for j in range(n):
            result[i][j]=C11[i][j]
            result[i][n+j]=C12[i][j]
            result[n+i][j]=C21[i][j]
            result[n+i][n+j]=C22[i][j]
    return result

# Stressen's matrix multiplication
# O(n^log7) ~ O(n^2.81)
def strassens_multiply(A,B,base):
    n=len(A)
    if(n<=base):
        return naive_multiply(A,B)
    # divide
    mid=(int)(n/2)
    A11=A[:mid, :mid]
    A12=A[:mid, mid:]
    A21=A[mid:, :mid]
    A22=A[mid:, mid:]
    B11=B[:mid, :mid]
    B12=B[:mid, mid:]
    B21=B[mid:, :mid]
    B22=B[mid:, mid:]

    # strassens 7 formulas
    multis1, m1=strassens_multiply(A11+A22,B11+B22,base)
    multis2, m2=strassens_multiply(A21+A22,B11,base)
    multis3, m3=strassens_multiply(A11,B12-B22,base)
    multis4, m4=strassens_multiply(A22,B21-B11,base)
    multis5, m5=strassens_multiply(A11+A12,B22,base)
    multis6, m6=strassens_multiply(A21-A11,B11+B12,base)
    multis7, m7=strassens_multiply(A12-A22,B21+B22,base)

    C11=m1+m4-m5+m7
    C12=m3+m5
    C21=m2+m4
    C22=m1+m3-m2+m6

    # combine
    multis=multis1+multis2+multis3+multis4+multis5+multis6+multis7
    return multis, np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    # return np.array(combine_matrix(C11,C12,C21,C22))