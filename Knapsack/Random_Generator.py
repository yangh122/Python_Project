import numpy as np


def random_instance(N,M,J,U):
    Seed = np.random.seed(12)
    C = np.zeros((J,N),dtype = int)
    A = np.zeros((M,N),dtype = int)
    B = np.zeros((M), dtype = int)


    for x in range(2500):
        for i in range(0,J):
            for j in range(0,N):
                C[i][j] = np.random.randint(1,40)

        for k in range(0,M):
            for i in range(0,N):
                A[k][i] = np.random.randint(1,40)

    for i in range (0,M):
        row_max = np.max(A[i])
        row_totals = sum(A[i])
        Halved_row_total = np.ceil(row_totals/2)
        row_max = np.max(A[i])
        if (row_max < Halved_row_total):
            B[i] = Halved_row_total
        else:
            B[i] = row_max
        Cost_minimized = np.negative(C)

    if (J==2):
        with open("inst_n5_m1_j2.txt", "w") as file:
            file.write(str(N))
            file.write("\n")
            np.savetxt(file, B, fmt = '%d')
            np.savetxt(file, Cost_minimized, fmt = '%d')
            np.savetxt(file, A,fmt = '%d')
    if (J==3):
        with open("inst_n5_m1_j3.txt", "w") as file:
            file.write(str(N))
            file.write("\n")
            np.savetxt(file, B, fmt = '%d')
            np.savetxt(file, Cost_minimized, fmt = '%d')
            np.savetxt(file, A,fmt = '%d')
    return

