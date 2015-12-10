/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <algorithm>
#include "autodiff.h"
#include "symbolic.h"
#include <ctime>
//#include <visualization/dagre.h>

//using namespace af;

//std::vector<float> input(100);

// Generate a random number between 0 and 1
// return a uniform number in [0,1].
//double unifRand()
//{
//    return rand() / double(RAND_MAX);
//}
//
//void testBackend()
//{
//    af::info();
//
//    af::dim4 dims(10, 10, 1, 1);
//
//    af::array A(dims, &input.front());
//    af_print(A);
//
//    af::array B = af::constant(0.5, dims, f32);
//    af_print(B);
//}
//
int main(int argc, char *argv[])
{
    af::dim4 dims(10, 10, 1, 1);
    autodiff::Graph graph = autodiff::Graph();

    auto c1 = graph.constant_node(af::constant(0.5, dims, f32));
    auto c2 = graph.constant_node(af::constant(1.5, dims, f32));

    auto i1 = graph.input_node();
    auto i2 = graph.input_node();

    auto s1 = graph.add(c1, i1);
    auto s2 = graph.add(c2, i2);

    //auto s4 = graph.neg(s2);
    auto s = graph.add(s1, s2);

    auto s_f = graph.add(s, i2);

    std::vector<autodiff::NodeId> params {i1 ,i2};
    {
        auto grads = graph.gradient(s_f, params);
        for(int i=0;i<grads.size();i++){
            std::cout << grads[i] << std::endl;
        }
        grads.push_back(s_f);
        for(int i=0;i<graph.nodes.size(); i++){
            auto node = graph.nodes[i];
//            std::cout << node->id << ", " << node->grad_level << std::endl;
        }
        autodiff::dagre::dagre_to_file("test_full.html", graph, grads);
    }

//
//
//    af::dim4 dims(10, 10, 1, 1);
//    diff::Graph graph = diff::Graph();
//    auto c1 = graph.create_constant_node(af::constant(0.5, dims, f32));
//    auto c2 = graph.create_constant_node(af::constant(1.5, dims, f32));
//    auto i1 = graph.create_input_node();
//    auto i2 = graph.create_input_node();
//    auto s1 = graph.mul(c1, i1);
//    auto s2 = graph.mul(c2, i2);
//    auto s4 = graph.neg(s2);
//    auto s = graph.add(s1, s4);
//    auto s_f = graph.mul(s, c2);
//    std::vector<diff::NodeId> params {i1 ,i2};
//    auto grads = graph.gradient(s_f, params);
//    for(int i=0;i<grads.size();i++){
//        std::cout << grads[i] << std::endl;
//    }
//    grads.push_back(s_f);
//    graph.print_to_file("tests.html", grads);
//    af::setBackend(AF_BACKEND_CPU);
//    int c = 2;
//    array randmat = af::randn(10000, 1000);
//    array randmat2 = af::randn(1000, 1000);
//    std::cout << randmat.numdims() << std::endl;
//    clock_t begin = clock();
//    auto result = af::matmul(randmat, randmat2);
//    af::eval(result);
//    clock_t end = clock();
//    double t =  double(end - begin) / CLOCKS_PER_SEC;
//    printf("Elapsed Time: %f sec\n", t);
//
//    const size_t N = 10;
//    std::vector<unsigned short> myvec;
//    int N = 2000000;
//    int M = 100;
//    int res = 0;
//    for(int i=0;i<N;i++){
//        myvec.push_back(i % 32);
//    }
//    std::clock_t start, finish;
//    long long m1, m2, m3;
//    auto f = [](unsigned short x){return (x+5)%17;};
//    for(int j=0;j<M;j++) {
//        start = std::clock();
//        for (std::vector<unsigned short>::const_iterator it = myvec.begin(); it != myvec.end(); it++) {
//            res += f(*it);
//        }
//        finish = std::clock();
//        m3 += finish - start;
//
//        start = std::clock();
//        for (int i = 0; i < myvec.size(); i++) {
//            res += f(myvec[i]);
//        }
//        finish = std::clock();
//        m1 += finish - start;
//
//        start = std::clock();
//        for (const unsigned short i : myvec) {
//            res += f(i);
//        }
//        finish = std::clock();
//        m2 += finish - start;
//
//    }
//    double f_m1 = ((double)(m1)) / (CLOCKS_PER_SEC);
//    double f_m2 = ((double)(m2)) / (CLOCKS_PER_SEC);
//    double f_m3 = ((double)(m3)) / (CLOCKS_PER_SEC);
//    std::cout << "(" << f_m1 << "," << f_m2 << "," << f_m3 << ")" << std::endl;
//    std::cout << res << std::endl;
//    auto t_a = symbolic::SymbolicMonomial<10, unsigned int>(2);
//    auto t_b = symbolic::SymbolicMonomial<10, unsigned int>(1);
//    auto t_c = 2*t_b*t_a*t_a;
//    std::cout << 5*t_c/(t_b*2*5) << std::endl;
//    auto t_p = t_c + t_a;
//    auto t_p2 = t_c - 2;
//    auto r = t_p * t_p2;
//    std::cout << t_p << std::endl;
//    std::cout << t_p2 << std::endl;
//    std::cout << r << std::endl;
//    std::cout << r / t_p << std::endl;
//
//    try {
//        printf("Trying CPU Backend\n");
//        af::setBackend(AF_BACKEND_CPU);
//        testBackend();
//    } catch (af::exception& e) {
//        printf("Caught exception when trying CPU backend\n");
//        fprintf(stderr, "%s\n", e.what());
//    }
//
//    try {
//        printf("Trying CUDA Backend\n");
//        af::setBackend(AF_BACKEND_CUDA);
//        testBackend();
//    } catch (af::exception& e) {
//        printf("Caught exception when trying CUDA backend\n");
//        fprintf(stderr, "%s\n", e.what());
//    }
//
//    try {
//        printf("Trying OpenCL Backend\n");
//        af::setBackend(AF_BACKEND_OPENCL);
//        testBackend();
//    } catch (af::exception& e) {
//        printf("Caught exception when trying OpenCL backend\n");
//        fprintf(stderr, "%s\n", e.what());
//    }

    return 0;
}