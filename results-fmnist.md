olia@olia-X555LJ:~/Documents/Programming_2021/Parallel_Programming_ece/ParallelProgrammingLabs/Lab4-NN$ gcc -o fmnist nn_fmnist-2.c -fopenmp -lm
olia@olia-X555LJ:~/Documents/Programming_2021/Parallel_Programming_ece/ParallelProgrammingLabs/Lab4-NN$ time ./fmnist 

X[60000][784], L1: 100, L2: 10,  epoch: 1000
Initialise fmnist data,W :  0.682202s
     MSE: 0.027583, time : 5.143038s
 100 MSE: 0.013489, time : 8.138292s
 200 MSE: 0.011950, time : 5.967105s
 300 MSE: 0.009765, time : 5.306875s
 400 MSE: 0.008898, time : 5.102918s
 500 MSE: 0.009068, time : 5.110120s
 600 MSE: 0.007795, time : 5.141652s
 700 MSE: 0.007475, time : 5.126513s
 800 MSE: 0.007152, time : 5.131338s
 900 MSE: 0.007161, time : 5.966761s

Test
Test MSE: 0.173984
Test Error: 1338 of 10000 (13.4 %)
    time : 0.475545s
Total time : 5602.718424s

real    93m22,764s
user    92m59,054s
sys     0m9,103s