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
    class FunctionBackend {
    private:
        void *handle;
    public:
        typedef std::vector<T> (*func_ptr)(std::vector<T>& inputs, std::vector<SharedPtr>& shared);
        class EvaluationFunction{
        public:
            std::vector<SharedPtr> shared_variables;
            std::vector <T> constant_variables;
            const func_ptr eval_func;

            EvaluationFunction(func_ptr eval_func):
                    eval_func(eval_func){};

            std::vector<T> eval(std::vector<T>& inputs){
                return eval_func(inputs, shared_variables);
            }
        };


        virtual void initialize() {

        };

        virtual void generate_source(std::string file_name, Graph graph,
                                     std::vector<Node> inputs,
                                     std::vector<Node> targets,
                                     Updates &updates) = 0;

        virtual void compile_file(std::string file_name, std::string dll_name) = 0;

        EvaluationFunction link_dll(std::string dll_name) {
            char *error_msg;
            handle = dlopen(("./" + dll_name).c_str(), RTLD_LAZY);
            if (!handle) {
                fputs(dlerror(), stderr);
                exit(1);
            }
            auto func_handle = (func_ptr) dlsym(handle, "eval_func");
            if ((error_msg = dlerror()) != NULL) {
                fputs(error_msg, stderr);
                exit(1);
            }
            return EvaluationFunction(func_handle);
        };

        void close() {
            dlclose(handle);
        }

        EvaluationFunction compile_function(std::string base_name,
                                  Graph graph,
                                  std::vector<Node> inputs,
                                  std::vector<Node> targets,
                                  Updates &updates) {
            clock_t start = clock();
            std::string source_name = base_name + ".cpp";
            std::string dll_name = base_name + ".so";
            generate_source(source_name, graph, inputs, targets, updates);
            compile_file(source_name, dll_name);
            EvaluationFunction function = link_dll(dll_name);
            clock_t end = clock();
            std::cout << "Compilation time: " << 1000 * (double(end - start)) / CLOCKS_PER_SEC << "ms" << std::endl;
            function.shared_variables = graph->shared_vars;
            return function;
        }

        void write_interface(std::ofstream &f) {
            f << "class InvalidInputShape : public std::exception {\n"
                         "public:\n"
                         "    size_t id;\n"
                         "    size_t expected[4];\n"
                         "    size_t given[4];\n"
                         "    std::string msg;\n"
                         "\n"
                         "    InvalidInputShape(size_t id,\n"
                         "                      size_t  expected[4],\n"
                         "                      size_t  given[4]) :\n"
                         "            id(id){\n"
                         "        for(int i=0;i<4;i++){\n"
                         "            this->expected[i] = expected[i];\n"
                         "            this->given[i] = given[i];\n"
                         "        }\n"
                         "        msg = \"The input node with id \" + std::to_string(id) + \" provided has incorrect shape.\\n\" +\n"
                         "              \"Expected:\" + std::to_string(expected[0]) + \", \" + std::to_string(expected[1]) + \", \"\n"
                         "              + std::to_string(expected[2]) + \", \" + std::to_string(expected[3]) + \", \" + \"\\n\" +\n"
                         "              \"Given:   \" + std::to_string(given[0]) + \", \" + std::to_string(given[1]) + \", \"\n"
                         "              + std::to_string(given[2]) + \", \" + std::to_string(given[3]) + \", \" + \"\\n\";\n"
                         "    };\n"
                         "\n"
                         "    const char *what() const throw() {\n"
                         "        return msg.c_str();\n"
                         "    }\n"
                         "};\n"
                         "\n"
                         "class SharedVariable{\n"
                         "public:\n"
                         "    size_t id;\n"
                         "    af::array value;\n"
                         "    SharedVariable():\n"
                         "            id(0),\n"
                         "            value(af::array())\n"
                         "    {};\n"
                         "    SharedVariable(size_t id, af::array value):\n"
                         "            id(id),\n"
                         "            value(value)\n"
                         "    {};\n"
                         "};\n"
                         "typedef std::shared_ptr<SharedVariable> SharedPtr;\n";
        }
    };
}
#endif //AUTODIFF_BACKENDS_BASE_H
