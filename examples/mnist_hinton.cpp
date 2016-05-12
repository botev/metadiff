
#include <arrayfire.h>
#include <sys/stat.h>
#include "metadiff.h"
#include "mnist.h"


namespace md = metadiff::api;
namespace dat = datasets;

struct program_args{
    af_backend backend;
    int iters;
    int burnout;
    int batch_size;
    int factor;
    bool burn;
    md::AfBackend::EvaluationFunction func;
    float * data_ptr;
    int * labels_ptr;
};

static program_args args;

void extract_args(int argc, char **argv);
void load_mnist(std::string folder);
void build_model(md::AfBackend& backend);
void run_model();

int main(int argc, char **argv) {
    // Some pre setup
    std::string name = "mnist_hinton";
    spdlog::set_pattern("*** [%H:%M:%S %z] [thread %t] %v ***");
    md::metadiff_sink->add_sink(std::make_shared<spdlog::sinks::stdout_sink_st>());

    extract_args(argc, argv);
    load_mnist(name);

    int batch_size_grid[3] = {1024, 2048, 4096};
    int factor_grid[3] = {1, 2, 4};

    // Set backend
    af::setBackend(args.backend);
    for(auto i = 0; i < 3; i++){
        for(auto j = 0; j < 3; j++){
            md::AfBackend backend = md::AfBackend(std::string(name));
            args.batch_size = batch_size_grid[i];
            args.factor = factor_grid[j];
            build_model(backend);
            std::cout << "batch_size=" << args.batch_size << " factor=" << args.factor << std::endl;
            args.burn = true;
            std::cout << "Running burnout" << std::endl;
            run_model();
            args.burn = false;
            std::cout << "Benchmarking..." << std::endl;
            std::cout << "Run time: "<< (af::timeit(run_model) / args.iters) << " seconds per iteration" << std::endl;
            backend.close();
            sleep(5);
        }
    }
}

void extract_args(int argc, char **argv){
    if(argc > 1){
        std::string cpu = "cpu";
        std::string opencl = "opencl";
        std::string cuda = "cuda";
        if(cpu.compare(argv[1]) == 0){
            args.backend = AF_BACKEND_CPU;
        } else if(opencl.compare(argv[1]) == 0) {
            args.backend = AF_BACKEND_OPENCL;
        } else if(cuda.compare(argv[1]) == 0) {
            args.backend = AF_BACKEND_CUDA;
        } else {
            std::cerr << "The first argument should be one of 'cpu', 'opencl' and 'cuda' - " << argv[1] << std::endl;
            exit(1);
        }
    } else {
        args.backend = AF_BACKEND_CPU;
    }
    if(argc > 2){
        std::istringstream ss(argv[2]);
        if(!(ss >> args.iters)) {
            std::cerr << "Invalid number " << argv[2] << '\n';
        }
    } else {
        args.iters = 20;
    }
    if(argc > 3){
        std::istringstream ss(argv[3]);
        if(!(ss >> args.burnout)) {
            std::cerr << "Invalid number " << argv[3] << '\n';
        }
    } else {
        args.burnout = 20;
    }
}

void load_mnist(std::string folder){
    mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
    dat::download_mnist(folder);
    args.data_ptr = new float[dat::MNIST_NUM_ROWS*dat::MNIST_NUM_COLS*dat::MNIST_NUM_IMAGES]{};
    args.labels_ptr = new int[dat::MNIST_NUM_IMAGES]{};
    dat::ReadTrainMNIST(folder, args.data_ptr, args.labels_ptr);
}

void build_model(md::AfBackend& backend){
    // Create graph
    auto graph = md::create_graph();
    graph->name = backend.name;
    // Batch size
    auto n = graph->get_new_symbolic_integer(); // a
    // Architecture
    int d[9] = {784, args.factor * 1000, args.factor * 500, args.factor * 250, args.factor * 30,
                args.factor * 250, args.factor * 500, args.factor * 1000, 784};
    // Test variable, should be removed from graph
    auto test = graph->constant_value(20);
    // Group names
    std::string layers[10] {"Inputs", "Encoder 1", "Encoder 2", "Encoder 3", "Encoder 4",
                            "Decoder 3", "Decoder 2", "Decoder 1", "Output Layer", "Objective"};
    // Input data
    graph->set_group(layers[0]);
    md::NodeVec inputs = {graph->matrix(md::dType::f32, {d[0], n}, "Input")};
    // Parameters
    std::vector<md::Node> params;
    for(int i=1;i<9;i++){
        graph->set_group(layers[i]);
        params.push_back(graph->shared_variable(af::randn(d[i], d[i-1], f32) / 100.0, "W_" + std::to_string(i)));
        params.push_back(graph->shared_variable(af::constant(float(0.0), d[i], 1, f32), "b_" + std::to_string(i)));
    }
    // First layer
    graph->set_group(layers[1]);
    auto h = md::tanh(md::dot(params[0], inputs[0]) + params[1]);
    // All layers except one
    for(int i=1;i<7;i++){
        graph->set_group(layers[i+1]);
        h = md::tanh(md::dot(params[2*i], h) + params[2*i+1]);
    }
    // Calculate only logits here
    graph->set_group(layers[8]);
    h = md::dot(params[14], h) + params[15];
    // Loss
    graph->set_group(layers[9]);
    auto error = md::binary_cross_entropy_logit(inputs[0], h);
    // Mean loss
    md::NodeVec loss = {error.sum() * graph->constant_value(1.0 / float(args.batch_size))};
    // Get grads
    auto grads = graph->gradient(loss[0], params);
    // Learning rate
    graph->set_group("SGD");
    auto learning_rate = graph->constant_value(0.01);
    // Set up sgd
    md::Updates updates;
    for(int i=0;i<params.size();i++){
        updates.push_back(std::pair<md::Node, md::Node>(params[i], params[i] - learning_rate * grads[i]));
    }
    std::string name = backend.name;
    name += dat::kPathSeparator + backend.name;
    // Optimize
    md::NodeVec new_inputs;
    md::NodeVec new_loss;
    md::Updates new_updates;
    md::Graph optimized =  graph->optimize(loss, updates, inputs,
                                           new_loss, new_updates, new_inputs);
    md::dagre_to_file(name + "_optim.html", optimized, new_updates);
    af::timer start = af::timer::start();
    args.func = backend.compile_function(optimized, new_inputs, new_loss, new_updates);
    std::cout << "Compile time: " << af::timer::stop(start) << " seconds" << std::endl;
}

void run_model(){
    std::vector<af::array> result;
    std::vector<af::array> data_inv;
    int n = args.burn ? args.burnout : args.iters;
    for(int i=0;i<n;i++){
        int ind = i % (dat::MNIST_NUM_IMAGES / args.batch_size);
        // Transfer data to device
        data_inv = {af::array(dat::MNIST_NUM_ROWS*dat::MNIST_NUM_COLS,
                              args.batch_size,
                              args.data_ptr + ind*args.batch_size*dat::MNIST_NUM_COLS*dat::MNIST_NUM_ROWS, afHost)};
        // Run function
        result = args.func.eval(data_inv);
    }
    for(auto i = 0; i < result.size(); i++){
        result[i].eval();
    }
    af::sync();
}