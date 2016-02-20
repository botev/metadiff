//
// Created by alex on 18/12/15.
//

#ifndef AUTODIFF_BACKENDS_BASE_H
#define AUTODIFF_BACKENDS_BASE_H

struct stat info;
#include "sys/stat.h"
namespace metadiff{
    /**
     * Abstract class for a backend
     */
    template<typename T>
    class FunctionBackend {
    protected:
        const char kPathSeparator =
        #ifdef _WIN32
                        '\\';
        #else
                        '/';
        #endif

        /**
         * Function to create a temporary directory and return its path
         * TODO - make this cross-platform
         */
        std::string get_and_create_temp_dir() {
            std::string path = std::tmpnam(nullptr);
            return path;
        };

        /**
         * Function to check if the folder exists and create it if not
         * TODO - make this cross-platform
         */
        void check_create_dir(std::string path) {
            if (stat(path.c_str(), &info) != 0) {
                logger()->debug() << name << "] Creating directory " << path;
                mkdir(path.c_str(), S_IRWXU);
            } else if (info.st_mode & S_IFDIR) {
                logger()->debug() << name << "] Directory " + path + " already exists";
            } else {
                CompilationFailed e = CompilationFailed("Directory " + path + " is a file");
                logger()->error() << name << "] " << e.msg;
                throw e;
            }
        }

        /**
         * Handle to the opened DLL
         */
        void *dll_handle;

        std::shared_ptr<spdlog::logger> logger() const {
            return metadiff::logger("backend");
        }

    public:
        /**
         * Name used for this backend
         */
        const std::string name;

        /**
         * Path to directory used for logging and storing outputs
         */
        std::string dir_path;

        FunctionBackend(std::string name, std::string dir_path) :
                name(name),
                dir_path(dir_path) { };

        FunctionBackend(std::string name) :
                name(name) {
            dir_path = get_and_create_temp_dir();
        };

        typedef std::vector<T> (*func_ptr)(std::vector<T> &inputs, std::vector<SharedPtr> &shared);

        class EvaluationFunction {
        public:
            std::vector<SharedPtr> shared_variables;
            std::vector<T> constant_variables;
            const func_ptr eval_func;

            EvaluationFunction(func_ptr eval_func) :
                    eval_func(eval_func) { };

            std::vector<T> eval(std::vector<T> &inputs) {
                return eval_func(inputs, shared_variables);
            }
        };

        /**
         * Any form of initialization required should be carried out here
         */
        virtual void initialize() { };

        /**
         * Generates the source code to the path specified
         */
        virtual void generate_source(std::string source_path,
                                     Graph graph,
                                     std::vector<Node> inputs,
                                     std::vector<Node> targets) = 0;

        /**
         * Compiles the source file to a dynamic library
         */
        virtual void compile_file(std::string source_path, std::string dll_path) = 0;

        EvaluationFunction link_dll(std::string dll_path) {
            logger()->debug() << name  << "] Linking file " << dll_path;
            char *error_msg;
            dll_handle = dlopen((dll_path).c_str(), RTLD_LAZY);
            if (!dll_handle) {
                CompilationFailed e = CompilationFailed("Error when opening DLL:" + std::string(dlerror()));
                logger()->error() << name << "] " << e.what();
                throw e;
            }
            auto func_handle = (func_ptr) dlsym(dll_handle, "eval_func");
            if ((error_msg = dlerror()) != NULL) {
                CompilationFailed e = CompilationFailed("Error when finding symbol:" + std::string(error_msg));
                logger()->error() << name << "] " << error_msg;
                throw e;
            }
            return EvaluationFunction(func_handle);
        };

        void close() {
            dlclose(dll_handle);
        }

        EvaluationFunction compile_function(Graph graph,
                                            std::vector<Node> inputs,
                                            std::vector<Node> targets,
                                            Updates &updates) {
            logger()->debug() << name << "] Compiling function to " << dir_path;
            check_create_dir(dir_path);
            // Set path for the source
            std::string source_path = dir_path;
            source_path += kPathSeparator;
            source_path += "src";
            check_create_dir(source_path);
            source_path += kPathSeparator;
            source_path += graph->name + ".cpp";

            // Generate the source
            graph->add_temporary_updates(updates);
            generate_source(source_path, graph, inputs, targets);
            graph->clear_temporary_updates();

            // Set path for the lib
            std::string dll_path = dir_path;
            dll_path += kPathSeparator;
            dll_path += "lib";
            check_create_dir(dll_path);
            dll_path += kPathSeparator;
            dll_path += graph->name + ".so";

            // Compile the source to the lib
            compile_file(source_path, dll_path);

            // Open the DLL
            EvaluationFunction function = link_dll(dll_path);

            // Set the shared variables
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
