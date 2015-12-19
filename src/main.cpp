
#include <arrayfire.h>

#include "metadiff.h"
namespace md = metadiff;
namespace sym = metadiff::symbolic;

void test(){
    af::array i1 = af::randn(100, 100);
    af::array i2 = af::randn(100, 100);
    af::array c = af::constant(1, 100);
    std::cout << "Parents dims: " << i1.dims() << "|" << i2.dims() << "|" << c.dims() <<  std::endl;
    af::gforSet(true);
    i1 += i2 * c;
    af::gforSet(false);
    double a[] {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    af::array t = af::array(2, 3, a);
    af_print(t);
    t += 1;
    af_print(t);
    std::cout << a[0] << ", " << a[1] << ", " << a[2] << std::endl;
}

int main(int argc, char *argv[])
{
//    test();
    auto graph = md::create_graph();
    graph->broadcast = md::ad_implicit_broadcast::WARN;
    // Batch size
    // Dim of input
    // Dim of hidden
    // Dim of output
    auto n = graph->get_new_symbolic_integer(); // a
    auto din = graph->get_new_symbolic_integer(); // b
    auto dh = graph->get_new_symbolic_integer(); // c
    auto dout = graph->get_new_symbolic_integer(); // d
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
    auto h1 = md::tanh(md::dot(Win, data_in) + bin);
    auto h2 = md::tanh(md::dot(Wh, h1) + bh);
    auto out = md::dot(Wout, h2) + bout;
    auto error = md::binary_cross_entropy_logit(data_t, out);
    auto error2 = error.reorder(1, 0);
    auto error3 = error2.flatten();
    auto sum_error = error3.sum();
    auto params = {Win, bin, Wh, bh, Wout, bout};
    auto grads = graph->gradient(sum_error, params);
    auto targets = grads;
    targets.push_back(sum_error);
    md::dagre::dagre_to_file("test_full.html", graph, targets);
    std::vector<md::Node> inputs{data_in, data_t,
                                 Win, Wh, Wout,
                                 bin, bh, bout};

    // Generate inputs
    int nv = 100; // a
    auto dinv = 784; // b
    auto dhv = 1000; // c
    auto doutv = 10; // d
    af::setSeed(100);
//    // Input data
//    auto data_inv = af::randn(dinv, nv);
//    auto data_tv = af::randu(doutv, nv);
//    // Parameters
//    auto Winv = af::randn(dhv, dinv);
//    auto Whv = af::randn(dhv, dhv);
//    auto Woutv = af::randn(doutv, dhv);
//    auto binv = af::randn(dhv);
//    auto bhv = af::randn(dhv);
//    auto boutv = af::randn(doutv);
//    std::vector<af::array> inputv{data_inv, data_tv,
//                                  Winv, Whv, Woutv,
//                                  binv, bhv, boutv};
    md::ArrayfireBackend backend("/opt/arrayfire-3/include");
    // Compile function
    auto train = backend.compile_function(graph, inputs, targets);

    // Run function
    long long time = 0;
    int N = 100;
    for(int i=0;i<N;i++){
        // Input data
        auto data_inv = af::randn(dinv, nv) / 10;
        auto data_tv = af::randu(doutv, nv);
        // Parameters
        auto Winv = af::randn(dhv, dinv);
        auto Whv = af::randn(dhv, dhv);
        auto Woutv = af::randn(doutv, dhv);
        auto binv = af::randn(dhv);
        auto bhv = af::randn(dhv);
        auto boutv = af::randn(doutv);
        std::vector<af::array> inputv{data_inv, data_tv,
                                      Winv, Whv, Woutv,
                                      binv, bhv, boutv};
        clock_t start = clock();
        auto result = train(inputv);
        clock_t end = clock();
        float *hv = result[6].host<float>();
        std::cout << "Value: " << hv[0] << std::endl;
        std::cout << "Elapsed time: " << 1000*(double(end - start))/CLOCKS_PER_SEC << "ms" << std::endl;
        time += (end-start);
    }
    backend.close();
    std::cout << "Mean Elapsed time: " << (1000/N)*(double (time))/CLOCKS_PER_SEC << "ms" << std::endl;
//    std::cout << "Saving" << std::endl;
    // Save parameters
//    af::saveArray("Win_grad", result[0], "test.b", true);
//    af::saveArray("bin_grad", result[1], "test.b", true);
//    af::saveArray("Wh_grad", result[2], "test.b", true);
//    af::saveArray("bn_grad", result[3], "test.b", true);
//    af::saveArray("Wout_grad", result[4], "test.b", true);
//    af::saveArray("bout_grad", result[5], "test.b", true);
//    af::saveArray("value", result[6], "test.b", true);
    return 0;
}