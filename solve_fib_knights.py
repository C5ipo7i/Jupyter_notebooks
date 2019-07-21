import numpy as np
import random
import math
import time
import copy
import sys

def a_fib(N):
    l = [0,1]
    while l[-2]+l[-1] < N:
        l.append(l[-2]+l[-1])
    return np.array(l[2:])

def return_left(N):
    max_num = N+N #exclusive
    fib_set = s_fib(max_num)
    assert N in fib_set
    if N % 2 == 0:
        return int(N / 2)
    else:
        fib_arr = a_fib(max_num)
        index = np.where(fib_arr == N)[0]
        print('index',index)
        # Check above and below for % 2 == 0
        if fib_arr[index+1] % 2 == 0:
            return int((fib_arr[index+1] / 2)[0])
        else:
            return int((fib_arr[index-1] / 2)[0])

def fibs_between_numbers(A,N,fibs):
    mask = np.logical_and(fibs > A,fibs <= N+A,fibs / A != 2)
    return fibs[mask]

def first_single(N,max_num,fib_arr):
    index = 0
    while index != N:
        num_fibs = fibs_between_numbers((N - index),N,fib_arr)
        if num_fibs.shape[0] > 1:
            pass
        else:
            return N-index,num_fibs[0]-(N-index)
        index += 1

# list is populated in reverse order
def solve_knights_left(N,chair_position):
    max_num = N+N #exclusive
    used = set()
    L = return_left(N)
    fib_arr = a_fib(max_num)
    end,pen_end = first_single(N,max_num,fib_arr)
    used.add(end)
    used.add(pen_end)
    used.add(L)
    solution_right = [pen_end,end]
    episode = 0
    solution = [L]
    # compute max possibility
    repeats = L
    next_num = L
    while len(solution) < chair_position: #N-2 for solving full list
        poss_fibs = fibs_between_numbers(next_num,N,fib_arr)
        poss_nums = np.subtract(poss_fibs,np.full(poss_fibs.shape[0],next_num,dtype=object))
#         print('solution',solution)
#         print('poss_fibs',poss_fibs)
#         print('poss_nums',poss_nums)
        for poss in reversed(poss_nums):
#             print('poss',poss)
            if poss in used:
                pass
            else:
                solution.append(poss)
                used.add(poss)
                next_num = poss
                break
        episode += 1
        if repeats == solution[-1]:
            # stuck
            break
        else:
            repeats = solution[-1]
#         if episode % 100000 == 0:
#             toc = time.time()
#             print("Episode {}, len(solution) = {}, time taken so far {} seconds".format(episode,len(solution),(toc-start)))
#             break
    return solution[-1]#solution+solution_right for returning full list

def main():
    # N = 144 # for testing purposes
    full_N = 99194853094755497
    chair_position_left = 1e+16
    # Because we are building the array from the reverse position
    # chair_position_right = int(full_N - chair_position_left)

    tic = time.time()
    solution = solve_knights_left(full_N,chair_position_left)
    toc = time.time()
    # Solution is returned as a single number instead of the full list, due to size
    print("The Solution {} took {} seconds".format(solution,(toc-tic)))

if __name__ == "__main__":
    main()