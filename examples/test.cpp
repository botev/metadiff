//
// Created by alex on 26/04/16.
//
#include "arrayfire.h"
#include "myreduce.h"
#include "stdio.h"
#include "iostream"

static const int n = 100;
static const int M = 200;
static const int D = 100;
static const int K = 50000;

void my_reduce(){
    af::array A = af::randn(M, D);
    af::array B = af::randn(K, D);
    af::array C = my_reduce_launcher(A, B);
    C.eval();
}

void gemm(){
    af::array A = af::randn(M, D);
    af::array B = af::randn(K, D);
    af::array C = af::matmul(A, B, AF_MAT_NONE, AF_MAT_TRANS);
    C.eval();
}

void rand_gen(){
    af::array A = af::randn(M, D);
    af::array B = af::randn(K, D);
    A.eval();
    B.eval();
}

void time1(){
    double rand_time = timeit(rand_gen);
    double my_reduce_time = timeit(my_reduce) - rand_time;
    double gemm_time = timeit(gemm) - rand_time;
    printf("Random generation took %g seconds\n", rand_time);
    printf("My reduce took %g seconds\n", my_reduce_time);
    printf("MatMul took %g seconds\n", gemm_time);
    printf("Ratio: %g\n",  my_reduce_time / gemm_time);
}

void time2(){
    af::array A = af::constant(0.0, M, D);
    af::array B = af::constant(0.0, K, D);
    af::array C = af::constant(0.0, M, K);
    af::timer start;
    for(int i=0;i<2*n;i++){
        A += af::randn(M, D);
        B += af::randn(K, D);
        A.eval();
        B.eval();
        if(i == n - 1){
            af::sync();
            start = af::timer::start();
        }
    }
    af::sync();
    double rand_time = af::timer::stop(start) / ((float) n);
    A = af::constant(0.0, M, D);
    B = af::constant(0.0, K, D);
    C = af::constant(0.0, M, K);
    for(int i=0;i<2*n;i++){
        A += af::randn(M, D);
        B += af::randn(K, D);
        C += my_reduce_launcher(A, B);
        C.eval();
        if(i == n - 1){
            af::sync();
            start = af::timer::start();
        }
    }
    af::sync();
    double my_reduce_time = af::timer::stop(start) / ((float) n) - rand_time;
    A = af::constant(0.0, M, D);
    B = af::constant(0.0, K, D);
    C = af::constant(0.0, M, K);
    for(int i=0;i<2*n;i++){
        A += af::randn(M, D);
        B += af::randn(K, D);
        C += af::matmul(A, B, AF_MAT_NONE, AF_MAT_TRANS);
        C.eval();
        if(i == n - 1){
            af::sync();
            start = af::timer::start();
        }
    }
    af::sync();
    double gemm_time = af::timer::stop(start) / ((float) n) - rand_time;
    printf("Random generation took %g seconds\n", rand_time);
    printf("My reduce took %g seconds\n", my_reduce_time);
    printf("MatMul took %g seconds\n", gemm_time);
    printf("Ratio: %g\n",  my_reduce_time / gemm_time);
}

int main(int argc, char *argv[])
{
    af::setBackend(AF_BACKEND_CUDA);
    if(argv[1] == '0'){
        time1();
    } else{
        time2();
    }
    return 0;
}

