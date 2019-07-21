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
def solve_knights(N,chair_position):
    max_num = N+N #exclusive
    used = set()
    start = time.time()
    fib_arr = a_fib(max_num)
    end,next_num = first_single(N,max_num,fib_arr)
    used.add(end)
    used.add(next_num)
    solution = [end,next_num]
    episode = 0
    repeats = next_num
    while len(solution) < chair_position+1: # N
        poss_fibs = fibs_between_numbers(next_num,N,fib_arr)
        poss_nums = np.subtract(poss_fibs,np.full(poss_fibs.shape[0],next_num,dtype=object))
        for poss in reversed(poss_nums):
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
    return solution[-1]

def main():
    N = 144 # for testing purposes
    full_N = 99194853094755497
    chair_position_left = 1e+16
    # Because we are building the array from the reverse position
    chair_position_right = int(full_N - chair_position_left)

    tic = time.time()
    solution = solve_knights(full_N,chair_position_right)
    toc = time.time()
    # Solution is returned as a single number instead of the full list, due to size
    print("The Solution {} took {} seconds".format(solution,(toc-tic)))

if __name__ == "__main__":
    main()