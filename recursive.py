# Recursive method - for both matrix of size n where n is power of 2
# O(n^log8) - O(n^3)
# dividing the matrix
def divide_matrix(M,sr,sc,er,ec):
    mat=[]
    for i in range(sr,er+1):
        temp=[]
        for j in range(sc,ec+1):
            temp.append(M[i][j])
        mat.append(temp)
    return mat

def add_matrix(A,B):
    return [[A[i][j]+B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def subt_matrix(A,B):
    return [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

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

def recursive_multiply(A,B,n):
    if n<=2:
        C=[[0,0],[0,0]]
        C[0][0]=A[0][0]*B[0][0] + A[0][1]*B[1][0]
        C[0][1]=A[0][0]*B[0][1] + A[0][1]*B[1][1]
        C[1][0]=A[1][0]*B[0][0] + A[1][1]*B[1][0]
        C[1][1]=A[1][0]*B[0][1] + A[1][1]*B[1][1]
        # global ms_r
        # ms_r+=8
        return C
    mid=(int)(n/2)
    # divide each matrix in four parts
    A11=divide_matrix(A,0,0,mid-1,mid-1)
    A12=divide_matrix(A,0,mid,mid-1,n-1)
    A21=divide_matrix(A,mid,0,n-1,mid-1)
    A22=divide_matrix(A,mid,mid,n-1,n-1)
    B11=divide_matrix(B,0,0,mid-1,mid-1)
    B12=divide_matrix(B,0,mid,mid-1,n-1)
    B21=divide_matrix(B,mid,0,n-1,mid-1)
    B22=divide_matrix(B,mid,mid,n-1,n-1)

    # multiply these as a 2x2 matrix
    C11=add_matrix(recursive_multiply(A11,B11,n//2),recursive_multiply(A12,B21,n//2))
    C12=add_matrix(recursive_multiply(A11,B12,n//2),recursive_multiply(A12,B22,n//2))
    C21=add_matrix(recursive_multiply(A21,B11,n//2),recursive_multiply(A22,B21,n//2))
    C22=add_matrix(recursive_multiply(A21,B12,n//2),recursive_multiply(A22,B22,n//2))

    # combine these four matrix 
    return combine_matrix(C11,C12,C21,C22)