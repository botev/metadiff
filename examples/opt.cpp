#include <arrayfire.h>
#include "metadiff.h"

// Some simple test in graph optimizations
// Add add_executable(opt.o examples/opt.cpp) in CMakeLists.txt

namespace md = metadiff::api;

struct program_args{
    af_backend backend;
    int iters;
    int burnout;
    int batch_size;
    int factor;
    bool burn;
    md::AfBackend func;
    float * data_ptr;
    int * labels_ptr;
};

static program_args args;

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

void build_model(){
    // Create graph
    auto graph = md::create_graph();

    graph->set_group("g1");

    //merge

    // auto x = graph->shared_variable(af::constant(float(2.0), 1, 1, f32), "x");
    // auto y = graph->shared_variable(af::constant(float(1.0), 1, 1, f32), "y");
    // auto z = md::dot(x + y, x + y);

    // const folding

    // auto x= graph->constant_value(1.5);
    // auto y= graph->constant_value(1.2);
    // auto z = graph->shared_variable(af::constant(float(2.0), 1, 1, f32), "z");

    // auto g = x+y;
    // auto q = g+y;
    // auto p = z+q;

    //const elimination

    // auto x = graph->constant_value(1);
    // auto y = graph->shared_variable(af::constant(float(2.0), 1, 1, f32), "y");
    // auto z = x*y;

    //neg neg
    // auto x = graph->shared_variable(af::constant(float(2.0), 1, 1, f32), "x");
    // auto y = x.neg();
    // auto z = y.neg();
    // auto p = graph->shared_variable(af::constant(float(2.0), 1, 1, f32), "p");
    // auto o = p.neg();
    // auto q = z*o;

    // sum scalar martix
    auto s1 = graph->shared_variable(af::constant(float(1.0), 1, 1, f32), "s1");
    auto s2 = graph->shared_variable(af::constant(float(2.0), 1, 1, f32), "s2");
    auto m1 = graph->shared_variable(af::constant(float(3.0), 2, 2, f32), "m1");
    // auto m2 = graph->shared_variable(af::constant(float(4.0), 2, 2, f32), "m2");
    
    //this is not created as nary
    //also it is elementwise .* looks like a dot product
    auto r = md::sum(s1 * s2 * m1);

    md::dagre_to_file("bin/mnist_hinton/opt_before.html", graph);

    graph->optimize();

    md::dagre_to_file("bin/mnist_hinton/opt_after.html", graph);

    // af::timer start = af::timer::start();
    // args.func.compile_function(graph);
    // std::cout << "Compile time: " << af::timer::stop(start) << " seconds" << std::endl;
}

int main(int argc, char **argv) {

    std::string name = "test_optimizations";
    md::metadiff_sink->add_sink(std::make_shared<spdlog::sinks::stdout_sink_st>());
    extract_args(argc, argv);
    af::setBackend(args.backend);
    args.func = md::AfBackend(std::string(name));

    build_model();

    args.func.close();

}