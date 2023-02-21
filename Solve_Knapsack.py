#imports
import time
import copy
import collections
import queue as Q
import numpy as np
import pandas as pd
import scipy as sp
import os

def SolveKnapsack(filename, method = 1):
    methodName = ''

    if method == 1:
        methodName = "BF"
        #TODO: Read and solve an instance via Brute-Force method
    
    #Read instance
    f = open(filename, 'r')

    read = f.readlines()
    total_lines = len(read)
    count = 0

    if(count == 0):
        num_of_items = int(read[0])
        count = count + 1
        record = count

    while(read[count][0] != '-'):
        count = count + 1

    constraint_RHS_array = np.empty((count - record), dtype = int)
    for i in range(count - record):
        constraint_RHS_array[i] = int(read[i+record])

    record = count

    while(read[count][0] == '-'):
        count = count + 1

    utility_array = np.empty([count - record, num_of_items], dtype = int)

    for j in range(count - record):
        utility_array[j] = [int(x) for x in read[record + j].split()]

    record = count

    constraint_coef_array = np.empty([total_lines - record, num_of_items], dtype = int)

    for k in range(total_lines - record):
        constraint_coef_array[k] = [int(x) for x in read[record + k].split()]

    start_time = time.time()

    #find feasible set (X) in decision space

    def bin_array(n): #generate all possible combination of binary x varaibles
        numbers = np.arange(2 ** n).reshape(2 ** n, 1)
        exponents = (2 ** np.arange(n))[::-1]
        return ((exponents & numbers) > 0).astype(int)


    all_possible = bin_array(num_of_items)
    feasible_set = np.zeros((0, num_of_items),dtype=int)
    
    #remove set from feasible set(X) if constraint is violated
    for i in range(all_possible.shape[0]):
        for j in range(constraint_coef_array.shape[0]):
            test_row =  all_possible[i] * constraint_coef_array[j]
            if (np.sum(test_row) <= constraint_RHS_array[j]):
                feasible_set = np.vstack((feasible_set, all_possible[i]))
    
    #find objective images
    z_initial = np.zeros((feasible_set.shape[0], utility_array.shape[0]), dtype=int)
    for i in range(feasible_set.shape[0]):
        for j in range(utility_array.shape[0]):
            z_initial[i][j] = np.sum(feasible_set[i] * utility_array[j])
    
    z_drop_duplicate = np.unique(z_initial, axis=0)
    print(z_drop_duplicate)

    def find_NDP_2D(arr):
        min_idx = np.argmin(arr[:,1])
        max_z1 = arr[min_idx][0]
        mask = arr[:,0] <=max_z1
        arr = arr[mask]

        min_indx = np.argmin(arr[:,0])
        max_z2 = arr[min_indx][1]
        mask = arr[:,1] <=max_z2
        arr = arr[mask]
        # Convert the array to a pandas DataFrame
        df = pd.DataFrame(arr, columns=['x', 'y'])
        # Find the nondominant points using groupby and transform
        is_dominated = df.groupby(['x'])['y'].transform('min') < df['y']
        # print(is_dominated)
        nondominant_points= df[~is_dominated]

        is_dominated = nondominant_points.groupby(['y'])['x'].transform('min') < nondominant_points['x']
        nondominant_points = nondominant_points[~is_dominated]
        
        # Convert the DataFrame of nondominant points to a NumPy array and return it
        return nondominant_points.values

    def find_NDP_3D(arr):
        min_indx = np.argmin(arr[:,0])
        max_1= arr[min_indx][1]
        max_2 = arr[min_indx][2]


        mask1 = arr[:,1] <=max_1
        mask2 = arr[:,2] <= max_2
        mask = mask1 | mask2
        arr = arr[mask]

        min_indx = np.argmin(arr[:,1])
        max_1= arr[min_indx][0]
        max_2 = arr[min_indx][2]

        mask1 = arr[:,0] <=max_1
        mask2 = arr[:,2] <= max_2
        mask = mask1 | mask2
        arr = arr[mask]

        min_indx = np.argmin(arr[:,2])
        max_1= arr[min_indx][0]
        max_2 = arr[min_indx][1]

        mask1 = arr[:,0] <=max_1
        mask2 = arr[:,1] <= max_2
        mask = mask1 | mask2
        arr = arr[mask]

        
        ndp = np.zeros((0, 3),dtype=int)

        for p in arr:
            is_ndp = True
            for q in arr:
                if np.all(p > q):
                    is_ndp = False
                    break
            if is_ndp:
                ndp = np.vstack((ndp, p))
        print(ndp)
        return ndp
   
    if(utility_array.shape[0] == 2):
        NDP = find_NDP_2D(z_drop_duplicate)
        INDEX = np.lexsort((-NDP[:,1], -NDP[:,0]))
        ndp_array = NDP[INDEX]
    if(utility_array.shape[0] == 3):
        NDP = find_NDP_3D(z_drop_duplicate)
        INDEX = np.lexsort((-NDP[:,2],-NDP[:,1], -NDP[:,0]))
        ndp_array = NDP[INDEX]

    total_time = time.time() - start_time


    # Output result
    ndp_filename = f'{"BF"}_NDP_{19}.txt'
    summary_filename = f'{"BF"}_SUMMARY_{19}.txt'
   
    #TODO: Export NDP and Summary files
    curr_dir = os.getcwd() + '/'
    np.savetxt(curr_dir + 'Solution.txt',ndp_array,delimiter='\t',newline='\n')
    
    summary_array = np.array([total_time, ndp_array.shape[0], 0])
    np.savetxt(curr_dir + 'Detail.txt', summary_array, delimiter='\t',newline='\n')
    return

SolveKnapsack("inst_n5_m1_j3.txt", 1)