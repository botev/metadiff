/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

//#include <arrayfire.h>
#include <algorithm>
#include "autodiff.h"
#include <ctime>

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
//    af::dim4 dims(1, 1, 1, 1);
//    auto c1_v = af::constant(0.5, dims, f32);
//    auto c2_v = af::constant(1.5, dims, f32);

    auto graph = autodiff::create_graph();
    graph->broadcast = autodiff::ad_implicit_broadcast::WARN;

//    auto c1 = graph->constant_node(af::constant(0.5, dims, f32));
//    auto c2 = graph->constant_node(af::constant(1.5, dims, f32));

//    auto c1 = graph->scalar(autodiff::ad_value_type::FLOAT);
//    auto c2 = graph->scalar(autodiff::ad_value_type::FLOAT);
    auto c1 = graph->constant_node(0.5);
    auto c2 = graph->constant_node(1.5);
    auto d = graph->tensor(autodiff::ad_value_type::FLOAT, {autodiff::SymInt(2), 3, 4, 5});

    auto i1 = graph->matrix(autodiff::ad_value_type::FLOAT);
    auto i2 = graph->vector(autodiff::ad_value_type::FLOAT, i1.shape()[0]);

    auto s1 = c1 + i1;
    auto s2 = c2 + i2;

    //auto s4 = graph->neg(s2);
    auto s = s1 + s2;

    auto s_f = (s + i2).sum();

    {
        auto grads = graph->gradient(s_f, {i1 ,i2});
        for(int i=0;i<grads.size();i++){
            std::cout << grads[i].id << std::endl;
        }
        grads.push_back(s_f);
        for(int i=0;i<graph->nodes.size(); i++){
            auto node = graph->nodes[i];
//            std::cout << node->id << ", " << node->grad_level << std::endl;
        }
        autodiff::dagre::dagre_to_file("test_full.html", graph, grads);
    }

    auto t_a = symbolic::SymbolicMonomial<10, unsigned int>(0);
    auto t_b = symbolic::SymbolicMonomial<10, unsigned int>(1);
    auto t_c = 2*t_b*t_a*t_a;
    auto t_p = t_c + t_a;
    auto t_p2 = t_c - 2;
    auto r = t_p * t_p2;
    return 0;
}