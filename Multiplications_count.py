import numpy as np
import matplotlib.pyplot as plt
import timeit
import time

def matrix_multiplication(A,B):
    if len(A[0]) != len(B):
        raise ValueError("Invalid matrix dimensions")
    result = [] # final result
    multis=0
    for i in range(len(A)):
        row = [] # the new row in new matrix
        for j in range(len(B[0])):
            product = 0 # the new element in the new row
            for v in range(len(A[i])):
                product += A[i][v] * B[v][j]
                multis+=1
            row.append(product) # append sum of product into the new row
        result.append(row) # append the new row into the final result

    return multis, np.array(result)

def pad_matrix(matrix):
    # Pad the matrix to the nearest power of 2 using zeros
    rows, cols = matrix.shape
    size = max(rows, cols)
    next_power_of_2 = 2 ** int(np.ceil(np.log2(size)))
    padded_matrix = np.zeros((next_power_of_2, next_power_of_2))
    padded_matrix[:rows, :cols] = matrix
    return padded_matrix

def strassen_matrix_multiply(A, B):
    # Pad matrices to the nearest power of 2
    A = pad_matrix(A)
    B = pad_matrix(B)

    n = A.shape[0]

    # Base case: If the matrices are small enough, use standard matrix multiplication
    if n <= 4:
        return matrix_multiplication(A, B)

    # Split matrices into submatrices
    half_size = n // 2
    A11 = A[:half_size, :half_size]
    A12 = A[:half_size, half_size:]
    A21 = A[half_size:, :half_size]
    A22 = A[half_size:, half_size:]

    B11 = B[:half_size, :half_size]
    B12 = B[:half_size, half_size:]
    B21 = B[half_size:, :half_size]
    B22 = B[half_size:, half_size:]

    # Recursive calls for subproblems
    m1,P1 = strassen_matrix_multiply(A11 + A22, B11 + B22)
    m2,P2 = strassen_matrix_multiply(A21 + A22, B11)
    m3,P3 = strassen_matrix_multiply(A11, B12 - B22)
    m4,P4 = strassen_matrix_multiply(A22, B21 - B11)
    m5,P5 = strassen_matrix_multiply(A11 + A12, B22)
    m6,P6 = strassen_matrix_multiply(A21 - A11, B11 + B12)
    m7,P7 = strassen_matrix_multiply(A12 - A22, B21 + B22)

    # Compute submatrices of the result
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    count=m1+m2+m3+m4+m5+m6+m7
    # Combine submatrices to form the result
    result = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return count,result


def alpha_tensor(A,B):

    A = pad_matrix(A)
    B = pad_matrix(B)

    n = A.shape[0]

    # Base case: If the matrices are small enough, use standard matrix multiplication
    if n <= 8:
        return matrix_multiplication(A, B)
    hs=n//2
    hhs=hs//2
    a11 = A[:hhs, :hhs]
    a12 = A[:hhs, hhs:hs]
    a13 = A[:hhs, hs:hs+hhs]
    a14 = A[:hhs, hs+hhs:]

    a21 = A[hhs:hs, :hhs]
    a22 = A[hhs:hs, hhs:hs]
    a23 = A[hhs:hs, hs:hs+hhs]
    a24 = A[hhs:hs, hs+hhs:]

    a31 = A[hs:hs+hhs, :hhs]
    a32 = A[hs:hs+hhs, hhs:hs]
    a33 = A[hs:hhs+hs, hs:hs+hhs]
    a34 = A[hs:hhs+hs, hs+hhs:]

    a41 = A[hs+hhs:, :hhs]
    a42 = A[hs+hhs:, hhs:hs]
    a43 = A[hhs+hs:, hs:hs+hhs]
    a44 = A[hhs+hs:, hs+hhs:]

    b11 = B[:hhs, :hhs]
    b12 = B[:hhs, hhs:hs]
    b13 = B[:hhs, hs:hs+hhs]
    b14 = B[:hhs, hs+hhs:]

    b21 = B[hhs:hs, :hhs]
    b22 = B[hhs:hs, hhs:hs]
    b23 = B[hhs:hs, hs:hs+hhs]
    b24 = B[hhs:hs, hs+hhs:]

    b31 = B[hs:hs+hhs, :hhs]
    b32 = B[hs:hs+hhs, hhs:hs]
    b33 = B[hs:hhs+hs, hs:hs+hhs]
    b34 = B[hs:hhs+hs, hs+hhs:]

    b41 = B[hs+hhs:, :hhs]
    b42 = B[hs+hhs:, hhs:hs]
    b43 = B[hhs+hs:, hs:hs+hhs]
    b44 = B[hhs+hs:, hs+hhs:]



    m1,h1 = alpha_tensor(a11,b13)
    m2,h2 = alpha_tensor((a11 +a31 + a33),(b11 + b31 + b33))
    m3,h3 = alpha_tensor((a11 + a31 + a34) ,(b12 + b42+b43))
    m4,h4 = alpha_tensor((a13 + a21 + a23) ,(b13 + b14 + b34))
    m5,h5 = alpha_tensor((a11 + a31),(b11 + b12 + b13 + b31 + b33 + b42 + b43))
    m6,h6 = alpha_tensor((a13 + a23),(b13 + b14+b32 + b33 + b34+b42+b43))
    m7,h7 = alpha_tensor((a14 +a43 +a44),(b31 + b33 + b41))
    m8,h8 = alpha_tensor((a14 +a41 +a44),(b13 + b14 + b44))
    m9,h9 = alpha_tensor((a13 +a23 + a24),(b32+b42 + b43))
    m10,h10 = alpha_tensor((a14+ a44),(b13 + b14+ b31 + b33 + b41 + b43 + b44))
    m11,h11 = alpha_tensor(a33,(b11 + b22 + b23 + b31 + b32))
    m12,h12 = alpha_tensor((a12+ a32 + a33),(b22 + b23 + b32))
    m13,h13 =alpha_tensor(a34,(b12 + b21 + b23 + b41 + b42))
    m14,h14 = alpha_tensor((a12+a32),(b21+b22+b23+b32+b41))
    m15,h15 = alpha_tensor((a12 + a32 + a34),(b21 + b23 + b41))
    m16,h16 = alpha_tensor(a21,(b12 + b14 + b22 + b23 + b34))
    m17,h17 = alpha_tensor((a12 + a21 + a22),(b12 + b22 + b23))
    m18,h18 = alpha_tensor((a12 + a22),(b12 + b22 + b23 + b24+b44))
    m19,h19 = alpha_tensor(a24,(b23 + b24 + b32 + b42 + b44))
    m20,h20 = alpha_tensor((a12 + a23 + a24 + a32 + a33),b32)
    m21,h21 = alpha_tensor((a12+ a22 + a24),(b23 + b24+b44))
    m22,h22 = alpha_tensor(a43,(b23 + b24 + b31 + b34 + b41))
    m23,h23 = alpha_tensor((a11 + a13 + a14 + a23 + a24+ a31 + a34),(b42 + b43))
    m24,h24 = alpha_tensor((a12 + a42 + a43),(b23 + b24+ b34))
    m25,h25 = alpha_tensor((a12+ a42),(b11 + b21 + b23 + b24 + b34))
    m26,h26 = alpha_tensor((a12+a41 + a42),(b11 + b21 + b23))
    m27,h27 = alpha_tensor(a14,b43)
    m28,h28 = alpha_tensor((a12 + a21 + a22 + a31 + a34),b12)
    m29,h29 = alpha_tensor((a12 + a21 + a23 + a42 + a43),b34)
    m30,h30 = alpha_tensor((a12 +a31 + a33 + a41 + a42),b11)
    m31,h31 = alpha_tensor(a41,(b11 + b14 + b21 + b23 + b44))
    m32,h32 = alpha_tensor((a12+ a32 + a34 + a43 + a44),b41)
    m33,h33 = alpha_tensor((a12 + a22 + a24 + a41 + a44),b44)
    m34,h34 = alpha_tensor((a21 + a31 + a41),(b11 + b12 + b14))
    m35,h35 =alpha_tensor((a12 + a21 + a22 + a32 + a33),(b22 + b23))
    m36,h36 = alpha_tensor((a12 + a24 +a32+ a43),(b23 + b24 + b32 + b41))
    m37,h37 = alpha_tensor((a12 + a21 + a33 + a42),(b11 + b22 + b23 + b34))
    m38,h38 = alpha_tensor((a22 +a32 + a42),(b21 + b22 + b24))
    m39,h39 = alpha_tensor(a12,b23)
    m40,h40 = alpha_tensor(a13,b33)
    m41,h41 = alpha_tensor((a11 + a13 + a14 + a21 + a23 + a41 + a44),(b13 + b14))
    m42,h42 = alpha_tensor((a12 +a32+ a34 + a41 + a42),(b21 + b23))
    m43,h43 = alpha_tensor((a24 +a34+ a44),(b41 + b42 + b44))
    m44,h44 = alpha_tensor((a23 + a33 + a43),(b31 + b32 + b34))
    m45,h45 = alpha_tensor((a11 + a13 + a14 + a31 + a33 + a43 + a44),(b31 + b33))
    m46,h46 = alpha_tensor((a12 + a22+ a34 + a41) ,(b12 + b21 + b23 + b44))
    m47,h47 = alpha_tensor((a12 + a22 + a24 + a42+ a43),(b23 + b24))
    C11 = h15 + h26 +h2 +h30 +h32 +h39 + h40 + h42 +h45 + h7
    C21 = h11+h12 +h14 +h20 +h22 +h24 +h25 + h29 +h35 +h36 +h37 +h38 +h44 +h47
    C31= h11+h12 +h14 +h15 +h26 +h30 +h39 + h42
    C41=h15 +h22 +h24 +h25+h26 +h32 +h39 + h42
    C12=h12 +h17 +h20 +h23 +h27 +h28 +h35 + h39 + h3 + h9
    C22 = h12 +h17 +h18 +h19 +h20 +h21 +h35 +h39
    C32=h12 + h13 +h14 +h15 + h17 + h28 +h35 +h39
    C42=h13 +h14 +h15 +h18 +h19 +h21 +h32 +h33 +h36 +h38 + h42 +h43 + h46 + h47
    C13=h1+h27 +h39 + h40
    C23 = h16 +h17 +h18 +h19 + h21 +h39 + h40 +h4+h6+ h9
    C33 = h11 + h12 + h13 +h14 + h15 + h1 + h2 +h39 +h3+h5
    C43=h10 +h22 +h24 +h25 + h26 +h27 +h31 + h39 + h7+h8
    C14=h1 + h21 + h24 + h29 + h33 + h39 + h41 + h47 +h4+h8
    C24 = h16 +h17+h18+h21 + h24 +h29 +h39 + h47
    C34 =h16 +h17 +h18 +h25 + h26 +h28 +h30 +h31 +h34 +h35 + h37 +h38 + h42 +h46
    C44 = h21 +h24 +h25 +h26 +h31 +h33 +h39 + h47
    result = np.vstack((np.hstack((C11, C12,C13,C14)), np.hstack((C21, C22,C23,C24)),np.hstack((C31,C32,C33,C34)),np.hstack((C41,C42,C43,C44))))
    count=m1+m2+m3+m4+m5+m6+m7+m8+m9+m10+m11+m12+m13+m14+m15+m16+m17+m18+m19+m20+m21+m22+m23+m24+m25+m26+m27+m28+m29+m30+m31+m32+m33+m34+m35+m36+m37+m38+m39+m40+m41+m42+m43+m44+m45+m46+m47
    return count,result

# Example usage with arbitrary matrix sizes
arr=[2,4,8,16,32,64,100,128,256,400,512,1024]
#arr=[4,8,16,32]
matrix_sizes=[]
execution_times=[]
execution_times_st=[]
execution_times_at=[]
naive_count=[]
strassen_count=[]
alpha_count=[]
for matrix_size in arr:
    matrix_sizes.append(matrix_size)
    # Generate random matrices of the specified size
    matrix1 = np.random.rand(matrix_size, matrix_size)
    matrix2 = np.random.rand(matrix_size, matrix_size)

    print("Matrix size: ",matrix_size)
    # Measure the time taken for matrix multiplication
    st=time.time()
    naive_multis, naiveAns = matrix_multiplication(matrix1, matrix2)
    naive_count.append(naive_multis)
    et=time.time()
    execution_times.append(et-st)
    print("Time taken by Naive: ",et-st)

    sst=time.time()
    strassen_multis, strassenAns=strassen_matrix_multiply(matrix1, matrix2)
    strassen_count.append(strassen_multis)
    sen=time.time()
    execution_times_st.append(sen-sst)
    print("Time taken by Strassen: ",sen-sst)

    sst=time.time()
    alpha_multis, alphaAns=alpha_tensor(matrix1, matrix2)
    alpha_count.append(alpha_multis)
    sen=time.time()
    execution_times_at.append(sen-sst)
    print("Time taken by Alpha Tensor: ",sen-sst)
    
# print(matrix_sizes)
# print(execution_times)
# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(matrix_sizes, execution_times, marker='o', linestyle='-',label='naive algo')
plt.plot(matrix_sizes, execution_times_st, marker='o', linestyle='--', label='strassen')
plt.plot(matrix_sizes, execution_times_at, marker='o', linestyle='-.', label='Alpha tensor')
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Matrix Multiplication Time vs. Matrix Size')
plt.grid(True)
plt.legend()
# Show the plot
plt.show()
# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

# result = strassen_matrix_multiply(A, B)
# print(result)

# for number of multiplications
plt.figure(figsize=(8, 6))
plt.plot(matrix_sizes, naive_count, marker='o', linestyle='-',label='naive algo')
plt.plot(matrix_sizes, strassen_count, marker='o', linestyle='--', label='strassen')
plt.plot(matrix_sizes, alpha_count, marker='o', linestyle='-.', label='Alpha tensor')
plt.xlabel('Matrix Size')
plt.ylabel('Number of multiplications')
plt.title('Matrix Size vs Number of multiplications')
plt.grid(True)
plt.legend()
# Show the plot
plt.show()