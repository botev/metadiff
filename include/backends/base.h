//
// Created by alex on 18/12/15.
//

#ifndef AUTODIFF_BACKENDS_BASE_H
#define AUTODIFF_BACKENDS_BASE_H

namespace metadiff{
    class MissingRequiredInput : public std::exception{
    public:
        std::vector<size_t> target_ids;
        size_t input_id;
        MissingRequiredInput(std::vector<size_t> targets, size_t input_id):
                target_ids(target_ids),
                input_id(input_id)
        {}

        MissingRequiredInput(std::vector<Node> targets, size_t input_id):
                input_id(input_id)
        {
            for(int i=0;i<targets.size();i++){
                this->target_ids.push_back(targets[i].id);
            }
        }

        const char* what() const throw(){
            std::string msg = "Missing required input " + std::to_string(input_id) + " for targets [";
            for(int i=0;i<target_ids.size();i++){
                msg += std::to_string(target_ids[i]);
                if(i<target_ids.size()-1){
                    msg += ", ";
                }
            }
            msg += "]";
            return msg.c_str();
        }
    };

    template<typename T>
    class FunctionBackend{
    private:
        void *handle;
    public:
        typedef std::vector<T> (*func_ptr)(std::vector<T>);
        virtual void initialize(){

        };
        virtual void generate_source(std::string file_name, Graph graph,
                                     std::vector<Node> inputs,
                                     std::vector<Node> targets) = 0;
        virtual void compile_file(std::string file_name, std::string dll_name) = 0;
        func_ptr link_dll(std::string dll_name){
            char *error_msg;
            handle = dlopen (("./" + dll_name).c_str(), RTLD_LAZY);
            if (!handle) {
                fputs (dlerror(), stderr);
                exit(1);
            }
            auto func_handle = (func_ptr) dlsym(handle, "eval_func");
            if ((error_msg = dlerror()) != NULL)  {
                fputs(error_msg, stderr);
                exit(1);
            }
            return func_handle;
        };

        void close(){
            dlclose(handle);
        }

        func_ptr compile_function(Graph graph,
                           std::vector<Node> inputs,
                           std::vector<Node> targets){
            clock_t start = clock();
            std::string source_name = "test.cpp";
            std::string dll_name = "test.so";
            generate_source(source_name, graph, inputs, targets);
            compile_file(source_name, dll_name);
            func_ptr function = link_dll(dll_name);
            clock_t end = clock();
            std::cout << "Compilation time: " << 1000*(double(end - start))/CLOCKS_PER_SEC << "ms" << std::endl;
            return function;
        }
    };
}
#endif //AUTODIFF_BACKENDS_BASE_H
