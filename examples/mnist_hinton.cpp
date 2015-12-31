
#include <arrayfire.h>
#include "metadiff.h"

namespace md = metadiff;
namespace sym = metadiff::symbolic;

int main(int argc, char **argv)
{
    std::string name = "mnist_hinton";
    // Default to CPU
    af_backend backend = AF_BACKEND_CPU;
    // Default batch size of 1000
    int batch_size = 1000;
    if(argc > 2){
        std::cerr << "Expecting two optional arguments - backend and batch size" << std::endl;
        exit(1);
    }
    if(argc > 1){
        std::string cpu = "cpu";
        std::string opencl = "opencl";
        std::string cuda = "cuda";
        if(cpu.compare(argv[1]) == 0){
            backend = AF_BACKEND_CPU;
        } else if(opencl.compare(argv[1]) == 0) {
            backend = AF_BACKEND_OPENCL;
        } else if(cuda.compare(argv[1]) == 0) {
            backend = AF_BACKEND_CUDA;
        } else {
            std::cout << (argv[1] == "opencl") << std::endl;
            std::cerr << "The first argument should be one of 'cpu', 'opencl' and 'gpu' - " << argv[1] << std::endl;
            exit(1);
        }
    }
    if(argc > 2){
        std::istringstream ss(argv[2]);
        if(!(ss >> batch_size)) {
            std::cerr << "Invalid number " << argv[2] << '\n';
        }
    }
    af::setBackend(backend);

    // Create graph
    auto graph = md::create_graph();
    graph->name = name;

    graph->broadcast = md::ad_implicit_broadcast::WARN;
    // Batch size
    auto n = graph->get_new_symbolic_integer(); // a
    // Real batch size
    // Architecture
    int d[9] = {784, 1000, 500, 250, 30, 250, 500, 1000, 784};
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
    // Calculate only logits here
    h = md::dot(params[14], h) + params[15];
    // Loss
    auto error = md::binary_cross_entropy_logit(data_in, h);
    // Mean loss
    auto loss = error.sum() * graph->constant_node(af::constant(1.0 / float(batch_size), 1));
    // Get grads
    auto grads = graph->gradient(loss, params);
    // Learning rate
    auto learning_rate = graph->constant_node(af::constant(0.0001, 1));

    // Set up sgd
    md::Updates updates;
    for(int i=0;i<params.size();i++){
        updates.push_back(std::pair<md::Node, md::Node>(params[i], params[i] - learning_rate * grads[i]));
    }

    // Print to file
    md::dagre::dagre_to_file(name + ".html", graph, {loss}, updates);

    // Create backend and compile function
    md::ArrayfireBackend md_backend("/opt/arrayfire-3/include", "/opt/arrayfire-3/lib");
    auto train = md_backend.compile_function(name, graph, {data_in}, {loss}, updates);

    // Run function
    long long time = 0;

    // Number of epochs
    int epochs = 100;
    // Number of epochs for burnout, to be discarded
    int burnout = 20;

    std::vector<af::array> data_inv = {af::randn(d[0], batch_size)};
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
    md_backend.close();

    std::cout << "Mean Elapsed time: " << 1000*(double (time))/(CLOCKS_PER_SEC*(epochs - burnout)) << "ms" << std::endl;
    return 0;
}