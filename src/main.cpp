
#include <arrayfire.h>

#include "metadiff.h"
namespace md = metadiff;
namespace sym = metadiff::symbolic;

void test()
{
    af::info();
    af_print(af::randu(5, 4));
}

void test_backend(){
    try {
        printf("Trying CPU Backend\n");
        af::setBackend(AF_BACKEND_CPU);
        test();
    } catch (af::exception& e) {
        printf("Caught exception when trying CPU backend\n");
        fprintf(stderr, "%s\n", e.what());
    }
    try {
        printf("Trying CUDA Backend\n");
        af::setBackend(AF_BACKEND_CUDA);
        test();
    } catch (af::exception& e) {
        printf("Caught exception when trying CUDA backend\n");
        fprintf(stderr, "%s\n", e.what());
    }
    try {
        printf("Trying OpenCL Backend\n");
        af::setBackend(AF_BACKEND_OPENCL);
        test();
    } catch (af::exception& e) {
        printf("Caught exception when trying OpenCL backend\n");
        fprintf(stderr, "%s\n", e.what());
    }
};

void print_mem_info(std::string name){
    size_t alloc_bytes,alloc_buffers,lock_bytes,lock_buffers;
    af::deviceMemInfo(&alloc_bytes,&alloc_buffers,&lock_bytes,&lock_buffers);
    std::cout << "Memory info " << name <<  std::endl;
    std::cout << "Allocated: " << alloc_bytes / 1024 << " KB" << std::endl;
    std::cout << "Buffers allocated: " << alloc_buffers << std::endl;
    std::cout << "In use: " << lock_bytes / 1024 << " KB" << std::endl;
    std::cout << "Buffers in use: " << lock_buffers << std::endl;
    return;
};

void test_dynamic_switch(){
    af::setBackend(AF_BACKEND_OPENCL);
    af::array a = af::randn(20, 20);
    af::array b = af::randn(20, 20);
    af_print(a);
    auto c  = af::matmul(a, b);
    auto d  = af::matmul(b, a);
    auto c_ptr = c.host<float>();
    auto c_dims = c.dims();
    auto d_ptr = d.host<float>();
    auto d_dims = d.dims();
    af::setBackend(AF_BACKEND_CPU);
    af::array c_cpu(c_dims, c_ptr, afDevice);
    af::array d_cpu(d_dims, d_ptr, afDevice);
    auto e_cpu = af::matmul(c_cpu, d_cpu);
    auto e_ptr = e_cpu.host<float>();
    auto e_dims = e_cpu.dims();
    af::setBackend(AF_BACKEND_OPENCL);
    af::array e(e_dims, e_ptr, afDevice);
    auto l = e + 5;
    af_print(l);
    delete[] c_ptr, d_ptr, e_ptr;
}

int main(int argc, char *argv[])
{
//    test_dynamic_switch();
//    test_backend();
//    print_mem_info("Test");
//    af::setBackend(AF_BACKEND_CPU);
//    af::array at = af::randu(20, 20);
//    af::array dt = af::constant(0.5, 1);
//    auto bt = at > 0.5;
//    std::cout << "SAD" << std::endl;
//    af::gforSet(true);
//    std::cout << "SAD" << std::endl;
//    af::array ct = af::select(bt, at, dt);
//    std::cout << "SAD" << std::endl;
//    af::gforSet(false);
//    std::cout << "SAD" << std::endl;
//    af::setSeed(20);

    af::setBackend(AF_BACKEND_CPU);
    auto graph = md::create_graph();
    graph->broadcast = md::ad_implicit_broadcast::WARN;
    // Batch size
    auto n = graph->get_new_symbolic_integer(); // a
    // Real batch size
    int nv = 1000; // a
    // Architecture
    int d[9] = {784, 1000, 500, 250, 30, 250, 500, 1000, 784};
//    int d[9] = {1536, 1536, 1536, 1536, 1536,  1536, 1536, 1536, 1536};
    // Input data
    auto data_in = graph->matrix(md::FLOAT, {d[0], n}, "Input");
    // Parameters
    std::vector<md::Node> params;
    for(int i=1;i<9;i++){
        params.push_back(graph->shared_var(af::randn(d[i], d[i-1]) / 100, "W_" + std::to_string(i)));
        params.push_back(graph->shared_var(af::constant(0.0, d[i]), "b_" + std::to_string(i)));
    }
    // Input Layer
    auto h = md::relu(md::dot(params[0], data_in) + params[1]);
    // All layers except one
    for(int i=1;i<7;i++){
        h = md::relu(md::dot(params[2*i], h) + params[2*i+1]);
    }
    // Calcualte only logits here
    h = md::dot(params[14], h) + params[15];
    // Loss
    auto error = md::binary_cross_entropy_logit(data_in, h);
    auto loss = error.sum() * graph->constant_node(af::constant(0.01, 1));
    // Get grads
    auto grads = graph->gradient(loss, params);

    auto learning_rate = graph->constant_node(af::constant(0.0001, 1));

    // Set up sgd
    md::Updates updates;
    for(int i=0;i<params.size();i++){
        updates.push_back(std::pair<md::Node, md::Node>(params[i], params[i] - learning_rate * grads[i]));
    }

    // Print to file
    md::dagre::dagre_to_file("test_full.html", graph, {loss}, updates);


    // Create backend and compile function
    md::ArrayfireBackend backend("/opt/arrayfire-3/include", "/opt/arrayfire-3/lib");
    auto train = backend.compile_function(graph, {data_in}, {loss}, updates);

    // Run function
    long long time = 0;
    // Number of epochs
    int epochs = 100;
    int burnout = 20;
    std::vector<af::array> data_inv = {af::randn(d[0], nv)};
    for(int i=0;i<epochs;i++){
        // Input data
        clock_t start = clock();
        auto result = train.eval(data_inv);
        clock_t end = clock();
        float *hv = result[0].host<float>();
        std::cout << "Value: " << hv[0] << std::endl;
        std::cout << "Elapsed time: " << 1000*(double(end - start))/CLOCKS_PER_SEC << "ms" << std::endl;
        if(i >= burnout) {
            time += (end - start);
        }
    }
    backend.close();
    std::cout << "Mean Elapsed time: " << 1000*(double (time))/(CLOCKS_PER_SEC*(epochs - burnout)) << "ms" << std::endl;
    std::cout << "Saving" << std::endl;
    return 0;
}