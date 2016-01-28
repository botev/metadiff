
#include <arrayfire.h>
#include <sys/stat.h>
#include "metadiff.h"
#include "mnist.h"
#include "iomanip"
#include <unistd.h>


namespace md = metadiff;
namespace sym = metadiff::symbolic;
namespace dat = datasets;


std::pair<double, double> run_md(int batch_size, int factor, int burnout, int epochs){
    // Download and load MNIST
    std::string name = "mnist_hinton";
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
    dat::download_mnist(name);
    float * data_ptr = new float[dat::MNIST_NUM_ROWS*dat::MNIST_NUM_COLS*dat::MNIST_NUM_IMAGES]{};
    int* labels_ptr = new int[dat::MNIST_NUM_IMAGES]{};
    dat::ReadTrainMNIST(name, data_ptr, labels_ptr);


    int period = 1;
    // Create graph
    auto graph = md::create_graph();
    graph->name = name;
    // If you want to be informed about broadcast change this
    graph->broadcast = md::ad_implicit_broadcast::QUIET;
    // Batch size
    auto n = graph->get_new_symbolic_integer(); // a
    // Real batch size
    // Architecture
    int d[9] = {784, factor * 1000, factor * 500, factor * 250, factor * 30,
                factor * 250, factor * 500, factor * 1000, 784};
    // Input data
    auto test = graph->constant_value(20);
    md::NodeVec inputs = {graph->matrix(md::FLOAT, {d[0], n}, "Input")};
    // Parameters
    std::vector<md::Node> params;
    for(int i=1;i<9;i++){
        params.push_back(graph->shared_var(af::randn(d[i], d[i-1], f32) / 100.0, "W_" + std::to_string(i)));
        params.push_back(graph->shared_var(af::constant(float(0.0), d[i], 1, f32), "b_" + std::to_string(i)));
    }
    // Input Layer
    auto h = md::tanh(md::dot(params[0], inputs[0]) + params[1]);
    // All layers except one
    for(int i=1;i<7;i++){
        h = md::tanh(md::dot(params[2*i], h) + params[2*i+1]);
    }
    // Calculate only logits here
    h = md::dot(params[14], h) + params[15];
    // Loss
    auto error = md::binary_cross_entropy_logit(inputs[0], h);
    // Mean loss
    md::NodeVec loss = {error.sum() * graph->constant_value(1.0 / float(batch_size))};
    // Get grads
    auto grads = graph->gradient(loss[0], params);
    // Learning rate
    auto learning_rate = graph->constant_value(0.01);
    // Set up sgd
    md::Updates updates;
    for(int i=0;i<params.size();i++){
        updates.push_back(std::pair<md::Node, md::Node>(params[i], params[i] - learning_rate * grads[i]));
    }
    name += dat::kPathSeparator + name;
    // Print to file
//    md::dagre::dagre_to_file(name + ".html", graph, loss, updates);
    // Optimize
    md::NodeVec new_inputs;
    md::NodeVec new_loss;
    md::Updates new_updates;
    md::Graph optimized =  graph->optimize(loss, updates, inputs,
                                           new_loss, new_updates, new_inputs);
//    std::cout << "Original:" << graph->nodes.size() << std::endl;
//    std::cout << "Optimized:" << optimized->nodes.size() << std::endl;
    md::dagre::dagre_to_file(name + "_optim.html", optimized, new_loss, new_updates);

    // Create backend and compile function
    md::ArrayfireBackend md_backend = md::ArrayfireBackend();
    auto train_org = md_backend.compile_function(name, graph, inputs, loss, updates);
    clock_t start = clock();
    double compile_time = 0;
    auto train_optim = md_backend.compile_function(name + "_optim", optimized, new_inputs, new_loss, new_updates);
    compile_time = ((double)(1000 * (clock() - start))) / ((double)(CLOCKS_PER_SEC));
    // Run function

    std::vector<af::array> result;
    std::vector<af::array> data_inv;
    float vals[epochs / period];
    double mean_time = 0;
    for(int i=0;i<epochs + burnout;i++){
        if(i == burnout){
            vals[0] = *result[0].host<float>();
            start = clock();
        }
        int ind = i % (dat::MNIST_NUM_IMAGES / batch_size);
        // Transfer data to device
        data_inv = {af::array(dat::MNIST_NUM_ROWS*dat::MNIST_NUM_COLS, batch_size,
                              data_ptr + ind*batch_size*dat::MNIST_NUM_COLS*dat::MNIST_NUM_ROWS, afHost)};
        result = train_optim.eval(data_inv);
        if(i >= burnout and (i + 1 - burnout) % period == 0) {
            vals[(i - burnout) / period] = *result[0].host<float>();
        }
    }
    mean_time = ((double)(1000 * (clock() - start))) / ((double)(CLOCKS_PER_SEC * epochs));
    md_backend.close();
    return std::pair<double, double>(mean_time, compile_time);
};


int main(int argc, char **argv)
{
    int batch_size_grid[3] = {1000, 5000, 10000};
    int factor_grid[3] = {1, 5, 10};
    // Default to CPU
    af_backend backend = AF_BACKEND_CPU;
    // Default repeats
    int repeats = 100;
    // Default burnout
    int burnout = 100;
    // Default number of epochs
    int epochs = 500;
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
            std::cerr << "The first argument should be one of 'cpu', 'opencl' and 'cuda' - " << argv[1] << std::endl;
            exit(1);
        }
    }
    if(argc > 2){
        std::istringstream ss(argv[2]);
        if(!(ss >> repeats)) {
            std::cerr << "Invalid number " << argv[2] << '\n';
        }
    }
    if(argc > 3){
        std::istringstream ss(argv[3]);
        if(!(ss >> burnout)) {
            std::cerr << "Invalid number " << argv[3] << '\n';
        }
    }
    if(argc > 4){
        std::istringstream ss(argv[4]);
        if(!(ss >> epochs)) {
            std::cerr << "Invalid number " << argv[4] << '\n';
        }
    }

    // Set backend
    af::setBackend(backend);
    af::array run_times = af::constant(0.0, 3, 3, repeats);
    af::array compile_times = af::constant(0.0, 3, 3, repeats);

    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            for(int r=0;r < repeats; r++){
                std::cout << "Running for batch size " << batch_size_grid[i]
                << " and factor " << factor_grid[j] << std::endl;
                std::pair<double, double> result = run_md(batch_size_grid[i], factor_grid[j], burnout, epochs);
                run_times(i, j, r) = result.first;
                compile_times(i, j, r) = result.second;
                std::cout << "Run: " << result.first << ", " << result.second << std::endl;
                af::deviceGC();
                usleep(5);
            }
        }
    }
    auto run_mean = af::mean(run_times);
    auto run_std = af::stdev(run_times);
    auto compile_mean = af::mean(run_times);
    auto compile_std = af::stdev(run_times);
    std::cout << "Run means:" << std::endl;
    af_print(run_mean);
    std::cout << "Run stds:" << std::endl;
    af_print(compile_std);
    std::cout << "Compile means:" << std::endl;
    af_print(compile_mean);
    std::cout << "Compile stds:" << std::endl;
    af_print(compile_std);

}
