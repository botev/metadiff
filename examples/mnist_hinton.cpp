
#include <arrayfire.h>
#include <sys/stat.h>
#include "metadiff.h"
#include "mnist.h"
#include "iomanip"


namespace md = metadiff;
namespace sym = metadiff::symbolic;
namespace dat = datasets;

int main(int argc, char **argv)
{
    // Download and load MNIST
    std::string name = "mnist_hinton";
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
    dat::download_mnist(name);
    float * data_ptr = new float[dat::MNIST_NUM_ROWS*dat::MNIST_NUM_COLS*dat::MNIST_NUM_IMAGES]{};
    int* labels_ptr = new int[dat::MNIST_NUM_IMAGES]{};
    dat::ReadTrainMNIST(name, data_ptr, labels_ptr);

    // Default to CPU
    af_backend backend = AF_BACKEND_CPU;
    // Default batch size of 1000
    int batch_size = 1000;
    // Default factor
    int factor = 1;
    // Default period
    int period = 1;
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
        if(!(ss >> batch_size)) {
            std::cerr << "Invalid number " << argv[2] << '\n';
        }
    }
    if(argc > 3){
        std::istringstream ss(argv[3]);
        if(!(ss >> factor)) {
            std::cerr << "Invalid number " << argv[3] << '\n';
        }
    }
    if(argc > 4){
        std::istringstream ss(argv[4]);
        if(!(ss >> period)) {
            std::cerr << "Invalid number " << argv[4] << '\n';
        }
    }
    std::cout << "Params: " << backend << ", " << batch_size << ", " << factor << ", " << period << std::endl;

    // Set backend
    af::setBackend(backend);
    // Transfer data to Arrayfire
    af::array data(dat::MNIST_NUM_IMAGES, dat::MNIST_NUM_ROWS*dat::MNIST_NUM_COLS, data_ptr, afHost);
    af::array l_in(dat::MNIST_NUM_IMAGES, labels_ptr, afHost);

    // Create graph
    auto graph = md::create_graph();
    graph->name = name;

    graph->broadcast = md::ad_implicit_broadcast::WARN;
    // Batch size
    auto n = graph->get_new_symbolic_integer(); // a
    // Real batch size
    // Architecture
    int d[9] = {784, factor * 1000, factor * 500, factor * 250, factor * 30,
                factor * 250, factor * 500, factor * 1000, 784};
    // Input data
    auto test = graph->constant_value(20);
    md::NodeVec inputs = {graph->matrix(md::FLOAT, {n, d[0]}, "Input")};
    // Parameters
    std::vector<md::Node> params;
    for(int i=1;i<9;i++){
        params.push_back(graph->shared_var(af::randn(d[i-1], d[i], f32) / 100.0, "W_" + std::to_string(i)));
        params.push_back(graph->shared_var(af::constant(float(0.0), 1, d[i], f32), "b_" + std::to_string(i)));
    }
    // Input Layer
    auto h = md::tanh(md::dot(inputs[0], params[0]) + params[1]);
    // All layers except one
    for(int i=1;i<7;i++){
        h = md::tanh(md::dot(h, params[2*i]) + params[2*i+1]);
    }
    // Calculate only logits here
    h = md::dot(h, params[14]) + params[15];
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
    md::dagre::dagre_to_file(name + ".html", graph, loss, updates);
    // Optimize
    md::NodeVec new_inputs;
    md::NodeVec new_loss;
    md::Updates new_updates;
    md::Graph optimized =  graph->optimize(loss, updates, inputs,
                                           new_loss, new_updates, new_inputs);
    std::cout << "Original:" << graph->nodes.size() << std::endl;
    std::cout << "Optimized:" << optimized->nodes.size() << std::endl;
    md::dagre::dagre_to_file(name + "_optim.html", optimized, new_loss, new_updates);

    // Create backend and compile function
    md::ArrayfireBackend md_backend = md::ArrayfireBackend();
    auto train_org = md_backend.compile_function(name, graph, inputs, loss, updates);
    auto train_optim = md_backend.compile_function(name + "_optim", optimized, new_inputs, new_loss, new_updates);

    // Run function
    long long time = 0;

    // Number of epochs for burnout, to be discarded
    int burnout = 100;
    // Number of epochs
    int epochs = 200;
    float *hv;
    clock_t start = clock();
    std::vector<af::array> result;
    std::vector<af::array> data_inv;
    for(int i=0;i<epochs + burnout;i++){
        if(i == burnout){
//            std::cout << "fetch" << std::endl;
            hv = result[0].host<float>();
            start = clock();
        }
        int ind = i % (dat::MNIST_NUM_IMAGES / batch_size);
        data_inv = {af::array(batch_size, dat::MNIST_NUM_ROWS*dat::MNIST_NUM_COLS, data_ptr + ind*batch_size, afHost)};
        result = train_optim.eval(data_inv);
//        std::cout << "I" << i << std::endl;
        if(i >= burnout and (i + 1 - burnout) % period == 0) {
//            std::cout << "fetch" << std::endl;
            hv = result[0].host<float>();
        }
    }
    time = (clock() - start);
    md_backend.close();
    std::cout << "Final Value: " << hv[0] << std::endl;
    std::cout << "Mean run time: " << std::setprecision(5) <<
            (1000*((double) (time)))/((double) (CLOCKS_PER_SEC*(epochs))) << "ms" << std::endl;
    return 0;
}
