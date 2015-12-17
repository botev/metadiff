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
//#include <ctime>
namespace md = metadiff;
namespace sym = metadiff::symbolic;

int main(int argc, char *argv[])
{

    auto graph = md::create_graph();
    graph->broadcast = md::ad_implicit_broadcast::WARN;
    // Batch size
    // Dim of input
    // Dim of hidden
    // Dim of output
    auto n = graph->get_new_symbolic_integer();
    auto din = graph->get_new_symbolic_integer();
    auto dh = graph->get_new_symbolic_integer();
    auto dout = graph->get_new_symbolic_integer();
    // Input data
    auto data_in = graph->matrix(md::FLOAT, {din, n}, "Input");
    auto data_t = graph->matrix(md::FLOAT, {dout, n}, "Targets");
    // Parameters
    auto Win = graph->matrix(md::FLOAT, {dh, din}, "Win");
    auto Wh = graph->matrix(md::FLOAT, {dh, dh}, "Wh");
    auto Wout = graph->matrix(md::FLOAT, {dout, dh}, "Wout");
    auto bin = graph->vector(md::FLOAT, dh, "bin");
    auto bh = graph->vector(md::FLOAT, dh, "bh");
    auto bout = graph->vector(md::FLOAT, dout, "bout");

    // Computation
    auto h1 = md::tanh(dot(Win, data_in) + bin);
    auto h2 = md::tanh(dot(Wh, h1) + bh);
    auto out = md::tanh(dot(Wout, h2) + bout);
    auto square_error = md::square(out - data_t).sum();
    auto params = {Win, bin, Wh, bh, Wout, bout};
    auto grads = graph->gradient(square_error, params);
    auto targets = grads;
    targets.push_back(square_error);
    md::dagre::dagre_to_file("test_full.html", graph, targets);
    return 0;
}