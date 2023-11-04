import numpy as np
import time
import matplotlib.pyplot as plot
from naive import naive_multiply
from recursive import recursive_multiply
from strassens import strassens_multiply

xaxis=[]
y1=[] # naive
y2=[] # recursive
y3=[] # strassen's
multis_r=[]
multis_s=[]
for i in range(2,11):
    Size=2**i
    xaxis.append(Size)
    # A=[[random.randint(1,9) for j in range(size)] for i in range(size)]
    # B=[[random.randint(1,9) for j in range(size)] for i in range(size)]
    A = np.random.randint(1,9,size=(Size,Size))
    B = np.random.randint(1,9,size=(Size, Size))
    start=time.time()
    multi,C1=naive_multiply(A,B)
    multis_r.append(multi)
    y1.append(time.time()-start)
    start=time.time()
    C2=recursive_multiply(A,B,Size)
    y2.append(time.time()-start)
    start=time.time()
    multi, C3=strassens_multiply(A,B,8)
    multis_s.append(multi)
    y3.append(time.time()-start)

# padding for random sizes
def padding(A):
    r,c = A.shape
    size = max(r,c)
    new_size = 2**int(np.ceil(np.log2(size)))
    result = np.zeros((new_size, new_size))
    result[:r,:c] = A
    return result

# sizes=[2,4,7,16,25,50,64,70,85,100,140,200,256,300,500]
# for Size in sizes:
#     xaxis.append(Size)
#     # A=[[random.randint(1,9) for j in range(size)] for i in range(size)]
#     # B=[[random.randint(1,9) for j in range(size)] for i in range(size)]
#     A = np.random.randint(1,9,size=(Size,Size))
#     B = np.random.randint(1,9,size=(Size,Size))
#     start=time.time()
#     C1=naive_multiply(A,B)
#     y1.append(time.time()-start)
#     # pad if not a power of 2
#     if np.ceil(np.log2(Size)) != np.floor(np.log2(Size)):
#         A=padding(A)
#         B=padding(B)
#     start=time.time()
#     Size=len(A)
#     C2=recursive_multiply(A,B,Size)
#     y2.append(time.time()-start)
#     start=time.time()
#     C3=strassens_multiply(A,B,2)
#     y3.append(time.time()-start)

plot.plot(xaxis,y1,xaxis,y2,xaxis,y3)
plot.title("Time Comparison between different matrix multiplication algorithms")
plot.xlabel("Size of Matrix")
plot.ylabel("Time Taken To multiply")
plot.legend(["Naive multiplication", "Recursive method", "Strassens Multiplication"])
plot.show()

# number of multiplication comparision
plot.plot(xaxis,multis_r,xaxis,multis_s)
plot.title("Multiplications in Naive approach and Strassen's approach")
plot.xlabel("Size of Matrix")
plot.ylabel("Number of multiplication")
plot.legend(["Naive Approach", "Strassens Approach"])
plot.show()

# print("Matrix A:")
# for r in A:
#     print(r)
# print("\nMatrix B:")
# for r in B:
#     print(r)
# print("\nMatrix C=AxB:")
# for r in C1:
#     print(r)
# for r in C2:
#     print(r)
# for r in C3:
#     print(r)
# for r in C4:
#     print(r)
# print(C1[0])
# print(C2[0])
# print(C3[0])