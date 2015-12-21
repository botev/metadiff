
#include <arrayfire.h>

#include "metadiff.h"
namespace md = metadiff;
namespace sym = metadiff::symbolic;

int main(int argc, char *argv[])
{
    af::setSeed(20);
    auto graph = md::create_graph();
    graph->broadcast = md::ad_implicit_broadcast::WARN;
    // Batch size
    auto n = graph->get_new_symbolic_integer(); // a
    // Real batch size
    int nv = 10000; // a
    // Architecture
//    int d[9] = {784, 1000, 500, 250, 30, 250, 500, 1000, 784};
    int d[9] = {784, 1536, 1536, 1536, 1536,  1536, 1536, 1536, 784};
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
    md::ArrayfireBackend backend("/opt/arrayfire-3/include");
    auto train = backend.compile_function(graph, {data_in}, {loss}, updates);

    // Run function
    long long time = 0;
    // Number of epochs
    int epochs = 1000;
    int burnout = 200;
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