//
// Created by alex on 18/12/15.
//

#ifndef AUTODIFF_BACKENDS_ARRAYFIRE_H
#define AUTODIFF_BACKENDS_ARRAYFIRE_H

namespace metadiff{
    namespace backend {
        using namespace exceptions;
#ifdef AFAPI
        class ArrayfireBackend : public FunctionBackend<af::array> {
        private:
            // types of variables to be generated
            enum GenType {
                AF_ARRAY = 0,
                STD_FLOAT = 1
            }; 

        public:
            std::string af_path;

            ArrayfireBackend(bool debug = false) :
                    FunctionBackend("ArrayFire", debug) {
                af_path = getenv("AF_PATH") ? getenv("AF_PATH") : "/opt/arrayfire-3";
                logger()->debug() << "af_path set to '" + af_path + "', debug flag is " + std::to_string(debug);
            };

            ArrayfireBackend(std::string dir_path, bool debug = false) :
                    FunctionBackend("ArrayFire", dir_path, debug) {
                af_path = getenv("AF_PATH") ? getenv("AF_PATH") : "/opt/arrayfire-3";
                logger()->debug() << "af_path set to '" + af_path + "', debug flag is " + std::to_string(debug);
            };

            ArrayfireBackend(std::string dir_path,
                             std::string af_path,
                             bool debug = false) :
                    FunctionBackend("ArrayFire", dir_path, debug),
                    af_path(af_path) {
                logger()->debug() << "af_path set to '" + af_path + "', debug flag is " + std::to_string(debug);
            };

            void compile(std::string source_dir, std::string target_dir, std::string graph_name) {
                std::string source_path = os::join_paths(source_dir, graph_name + ".cpp");
                std::string dll_path = os::join_paths(target_dir, graph_name + ".so");
                logger()->debug() << "Compiling file " << source_path << " to " << dll_path;
                std::string log_path = source_path + ".log";
                std::string command = "MKL_NUM_THREADS=4 g++ -O3 -Wall -shared -fPIC -std=c++11 -laf ";
                command += "-Werror=return-type -Wno-unused-variable -Wno-narrowing ";
                command += " -I" + os::join_paths(af_path, "include");
                command += " -L" + os::join_paths(af_path, "lib");
                command += " -o " + dll_path + " " + source_path;
                command += " > " + log_path + " 2>&1";
                logger()->debug() << "Compile command: " << command;
                int response = system(command.c_str());
                if (response != 0) {
                    std::ifstream log_file(log_path);
                    std::string err_msg((std::istreambuf_iterator<char>(log_file)),
                                        std::istreambuf_iterator<char>());
                    auto err = CompilationFailed("Bad compilation response: " + std::to_string(response) +
                                                 ", command output: " + err_msg);
                    logger()->error() << err.msg;
                    throw err;
                }
                return;
            }

            func_ptr link(std::string target_dir,
                                    std::string graph_name) {
                logger()->debug() << os::join_paths(target_dir, graph_name + ".so");
                return link_dll(os::join_paths(target_dir, graph_name + ".so"), "eval_func");
            }

            void generate_source(std::string source_dir,
                                 Graph graph,
                                 std::vector<Node> inputs,
                                 std::vector<Node> targets) {
                std::string source_path = os::join_paths(source_dir, graph->name + ".cpp");
                logger()->trace() << "Generating source file " << source_path;
                std::ofstream f;
                f.open(source_path);

                // Print disclaimer
                f << "// Auto generated by Metadiff\n// Please do not edit\n\n";

                // Print includes
                f << "#include \"vector\"\n"
                        "#include \"iostream\"\n"
                        "#include \"memory\"\n"
                        "#include <exception>\n"
                        "#include <arrayfire.h>\n";
                f << "\n";

                // Write the interface to Shared Variables and InputShapeExceptions
                write_interface(f);
                write_af_interface(f);

                // Print a helper function for memory info
//                f << "void print_mem_info(std::string name){\n"
//                        "\tsize_t alloc_bytes,alloc_buffers,lock_bytes,lock_buffers;\n"
//                        "\taf::deviceMemInfo(&alloc_bytes,&alloc_buffers,&lock_bytes,&lock_buffers);\n"
//                        "\tstd::cout << \"Memory info\" << name << std::endl;\n"
//                        "\tstd::cout << \"Allocated: \" << alloc_bytes / 1024 << \" KB\" << std::endl;\n"
//                        "\tstd::cout << \"Buffers allocated: \" << alloc_buffers << std::endl;\n"
//                        "\tstd::cout << \"In use: \" << lock_bytes / 1024 << \" KB\" << std::endl;\n"
//                        "\tstd::cout << \"Buffers in use: \" << lock_buffers << std::endl;\n"
//                        "\treturn;\n"
//                        "};\n\n";

                // Print the function interface
                f << "extern \"C\" std::vector<af::array> "
                        "eval_func(std::vector<af::array>& inputs, "
                        "std::vector<SharedPtr>& shared_vars){\n";
                // Use the gfor
                f << "\t// Set up automatic broadcasting\n";
                f << "\taf::gforSet(true);\n";

                // Check all of the required inputs are provided
                for (size_t i = 0; i < graph->nodes.size(); i++) {
                    if (graph->nodes[i]->node_type == core::INPUT and
                        graph->nodes[i]->op->name != "Shared") {
                        for (size_t j = 0; j <= inputs.size(); j++) {
                            if (j == inputs.size()) {
                                auto err = MissingRequiredInput(targets, inputs, graph->nodes[i]);
                                logger()->error() << err.msg;
                                throw err;
                            }
                            if (inputs[j]->id == i) {
                                break;
                            }
                        }
                    }
                }

                // An expression table for all nodes
                std::vector<std::string> expression_table(graph->nodes.size(), "Undefined");
				// keep track of the generated types of each node
                std::vector<GenType> node_types(graph->nodes.size(), GenType::AF_ARRAY);
                // keep track of each tag
                unordered_set<int> tags;

                // consider inlined and inplace optimizations here

                // Loop over all nodes and calculate their expressions
                // as well as write anything that is not inlined
                f << "\n\t// Calculate all of the computation nodes\n";
                for (size_t i = 0; i < graph->nodes.size(); i++) {
                    std::shared_ptr<NodeInternal> node = graph->nodes[i];

                    std::string expression = node_expression(node, expression_table);
                    if (graph->nodes[i]->execution.inlined) {
                        expression_table[i] = expression;
                    } else {
                        if (debug) {
                            f << "\tstd::cout << \"Calculating node '" << i << "'\" << std::endl;\n";
                        }

                        // TODO this should be properly done for all scalar types
                        // The code generated is af::array node_index = <expression>;
                        if (Node(graph->nodes[i]).is_constant() and Node(graph->nodes[i]).is_scalar()) {
                            f << "\tfloat ";
                            node_types[i] = GenType::STD_FLOAT;
                        }
                        else if (graph->nodes[i]->execution.inplace){
                            // only create new array if the node is has no inplace node
                            f << "\t";
                        }
                        else if (graph->nodes[i]->execution.tag != -1 and 
                            tags.find(graph->nodes[i]->execution.tag) != tags.end()) {
                            f << "\t";
                        }
                        else {
                            f << "\taf::array ";
                            tags.insert(graph->nodes[i]->execution.tag);
                        }

                        int nodeNum = i;
                        // reuse the inplace node
                        if (graph->nodes[i]->execution.inplace) {
                            nodeNum = graph->nodes[i]->execution.inplace->id;
                        }

                        if (graph->nodes[i]->execution.tag != -1) {
                            nodeNum = graph->nodes[i]->execution.tag;
                        }

                        f << "node_" << nodeNum << " = " << expression << ";\n";
                        expression_table[i] = "node_" + std::to_string(nodeNum);

                        if (debug) {
                            // float has no dims()

                            if (node_types[i] != GenType::STD_FLOAT)
                                f << "\tstd::cout << \"Node size:\" << node_" << nodeNum << ".dims() << std::endl;\n";
                            if (graph->nodes[i]->execution.inplace) {
                                f << "\tstd::cout << \"node " << i << " is inplace node "<< nodeNum << "\"<< std::endl;\n";
                            }
                        }
                    }
                }

                // Update all of the shared_variables
                f << "\n\t// Update all shared variables\n";
                for (size_t i = 0; i < graph->updates.size(); i++) {
                    if (debug) {
                        f << "\tstd::cout << \"Calculating update '" << i << "'\" << std::endl;\n";
                    }
                    print_update(f, graph->updates[i], expression_table);
                }
                for (size_t i = 0; i < graph->temporary_updates.size(); i++) {
                    if (debug) {
                        f << "\tstd::cout << \"Calculating update '" << i << "'\" << std::endl;\n";
                    }
                    print_update(f, graph->temporary_updates[i], expression_table);
                }

                // Disable the automatic broadcasting
                // TODO Decide whether to include this? Maybe have to check what it was before
                // f << "\taf::gforSet(false);";

                // Write all of the output nodes as the return statement
                f << "\n\t// Write all of the output nodes in correct order\n";
                f << "\treturn {";
                for (size_t i = 0; i < targets.size(); i++) {
                    if (i < targets.size() - 1) {
                        f << expression_table[targets[i]->id] << ", ";
                    } else {
                        f << expression_table[targets[i]->id] << "};\n";
                    }
                }
                // Close the function block and the file
                f << "}\n";
                f.close();
            }

            void print_operator(std::string name, std::vector<size_t> p1,
                                std::vector<size_t> args) {

            }

            void print_update(std::ofstream &f, std::pair<Node, Node> graph_update,
                              std::vector<std::string> &expression_table) {
                std::shared_ptr<op::SharedInput> cast_op = std::static_pointer_cast<op::SharedInput>(graph_update.first->op);
                size_t shared_id = cast_op->var->id;
                Node update = graph_update.second;

                if (not update->execution.inlined or
                    (update->op->name != "Add" and update->op->name != "Mul")) {
                    f << "\t" << shared_value(shared_id)  << " = "
                    << expression_table[update->id] << ";\n";
                } else {
                    // This part is for updates of the form
                    // x = x + ..., x - ..., x * ... or x / ...
                    // I try to merge this cases into one
                    // Note that all of this operations are either and Add or a Mul
                    std::string pos_char, neg_char, pos_name, neg_name;
                    std::string prefix = "";
                    if (update->op->name == "Add") {
                        pos_char = "+";
                        neg_char = "-";
                        neg_name = "Neg";
                    } else {
                        pos_char = "*";
                        neg_char = "/";
                        neg_name = "Div";
                    }
                    // First we need to check if the shared variable is in this operator
                    // Index of the shared_variable if it is present in the operator
                    int index = -1;
                    // All other parents are negative operator
                    // This is required in order to distinguish between += and -=, *= and /=
                    bool all_neg = true;
                    NodeVec parents = update->op->get_parents();
                    for (int i = 0; i < parents.size(); i++) {
                        if (parents[i]->op->name == "Shared") {
                            std::shared_ptr<op::SharedInput> cast_op2 = std::static_pointer_cast<op::SharedInput>(
                                    parents[i]->op);
                            if (cast_op2->var->id == shared_id) {
                                index = i;
                            }
                        } else if (parents[i]->op->name != neg_name) {
                            all_neg = false;
                        }
                    }
                    if (index == -1) {
                        // If the shared variable is not in the operator than it is a standard syntax
                        f << "\t" << shared_value(shared_id) << " = "
                        << expression_table[update->id] << ";\n";
                    } else {
                        // Remove the shared variable from the parents
                        parents.erase(parents.begin() + index);
                        // If all are negative we make the prefix the neg_char, otherwise the pos_char
                        if (all_neg) {
                            prefix = neg_char;
                        } else {
                            prefix = pos_char;
                        }
                        f << "\t" << shared_value(shared_id) << prefix << "=";
                        if (all_neg) {
                            // If all are negative we need the grand parents rather than the parents
                            for (size_t i = 0; i < parents.size(); i++) {
                                size_t id = parents[i]->op->get_parents()[0]->id;
                                if (i == 0) {
                                    f << " " << expression_table[id];
                                } else {
                                    f << " " << pos_char << " " << expression_table[id];
                                }
                                if (i < parents.size() - 1) {
                                    f << pos_char;
                                }
                            }
                            f << ";\n";
                        } else {
                            for (size_t i = 0; i < parents.size(); i++) {
                                // If the parent is a negative operator we need to write its grand parent
                                // with a neg_char
                                if (i == 0) {
                                    f << " " << expression_table[parents[i]->id];
                                } else if (parents[i]->op->name != neg_name) {
                                    f << " " << pos_char << " " << expression_table[parents[i]->id];
                                } else {
                                    size_t id = parents[i]->op->get_parents()[0]->id;
                                    f << " " << neg_char << " " << expression_table[id];
                                }
                            }
                            f << ";\n";
                        }

                    }
                }
            }


            std::string node_expression(Node node, std::vector<std::string> &expression_table) {
                auto node_in = node;
                auto op_name = node_in->op->name;
                auto parents = node_in->op->get_parents();
                auto args = node_in->op->get_arguments();
                auto children = node_in->children;

                // Constant operators
                if (op_name == "MakeConst") {
                    return expression_table[node_in->id];
                }
                if (op_name == "Eye") {
                    // TODO actually have to implement symbolics
                    return "NotImplemented";
                }
                if (op_name == "Zeros") {
                    if (node.is_scalar()) {
                        return "0.0";
                    } else {
                        // TODO actually have to implement symbolics
                        return "NotImplemented";
                    }
                }
                if (op_name == "Ones") {
                    if (node.is_scalar()) {
                        return "1.0";
                    } else {
                        // TODO
                        return "NotImplemented";
                    }
                }
                if (op_name == "ConstValue") {
                    std::shared_ptr<op::ConstantValue> cast_op = std::static_pointer_cast<op::ConstantValue>(node_in->op);
                    if (node.is_scalar()) {
                        // TODO correctly do this
                        return std::to_string(cast_op->value);
                    } else {
                        // TODO
                        return "NotImplemented";
                    }
                }
                if (op_name == "Seq") {
                    // TODO
                    return "NotImplemented";
                }

                // Base operators
                if (op_name == "Input") {
                    return "inputs[0]";
                }
                if (op_name == "Shared") {
                    std::shared_ptr<op::SharedInput> cast_op2 = std::static_pointer_cast<op::SharedInput>(node_in->op);
                    return  shared_value(cast_op2->var->id);
                }
                if (op_name == "Alias") {
                    return expression_table[parents[0]->id];
                }
                if (op_name == "Broadcast") {
                    bool not_supported = false;
                    for (size_t i = 0; i < children.size(); i++) {
                        auto name = children[i]->op->name;
                        if (name != "Add" and name != "Mul"
                            and name != "Neg" and name != "Div") {
                            not_supported = true;
                            break;
                        }
                    }
                    if (not_supported) {
                        // For operators where this is not supported we have to use af::tile()
                        std::string expression = "af::tile(" + expression_table[parents[0]->id] + ", ";
                        for (int i = 0; i < 4; i++) {
                            if (node_in->shape[i] != parents[0]->shape[i]) {
                                expression += node_in->shape[i].to_string_with_star();
                            } else {
                                expression += "1";
                            }
                            if (i < 3) {
                                expression += ", ";
                            }
                        }
                        return expression + ")";
                    } else {
                        return expression_table[parents[0]->id];
                    }
                }
                if (op_name == "Add") {
                    std::string expression = expression_table[parents[0]->id];
                    for (int i = 1; i < parents.size(); i++) {
                        if (parents[i]->op->name == "Neg") {
                            expression +=
                                    " - " + expression_table[parents[i]->op->get_parents()[0]->id];
                        } else {
                            expression += " + " + expression_table[parents[i]->id];
                        }
                    }
                    return "(" + expression + ")";
                }
                if (op_name == "Neg") {
                    return "(-" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Mul") {
                    std::string expression = expression_table[parents[0]->id];
                    for (int i = 1; i < parents.size(); i++) {
                        if (parents[i]->op->name == "Div") {
                            expression +=
                                    " / " + expression_table[parents[i]->op->get_parents()[0]->id];
                        } else {
                            expression += " * " + expression_table[parents[i]->id];
                        }
                    }
                    return expression;
                }
                if (op_name == "Div") {
                    return "(1.0/" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Sum") {
                    auto axes = dynamic_cast<op::Sum *>(node_in->op.get())->axes;
                    if (Node(node).is_scalar()) {
                        return "af::sum(af::flat(" + expression_table[parents[0]->id] + "))";
                    } else {
                        std::string expression = expression_table[parents[0]->id];
                        for (size_t i = 0; i < axes.size(); i++) {
                            expression = "af::sum(" + expression + ", " + std::to_string(axes[i]) + ")";
                        }
                        return expression;
                    }
                }
                // Logical operators
                if (op_name == "Not") {
                    return "!" + expression_table[parents[0]->id];
                }
                if (op_name == "Gt") {
                    return expression_table[parents[0]->id] + " > " +
                           expression_table[parents[1]->id];
                }
                if (op_name == "Ge") {
                    return expression_table[parents[0]->id] + " >= " +
                           expression_table[parents[1]->id];
                }
                if (op_name == "Lt") {
                    return expression_table[parents[0]->id] + " < " +
                           expression_table[parents[1]->id];
                }
                if (op_name == "Lte") {
                    return expression_table[parents[0]->id] + " <= " +
                           expression_table[parents[1]->id];
                }
                if (op_name == "Eq") {
                    return expression_table[parents[0]->id] + " == " +
                           expression_table[parents[1]->id];
                }
                if (op_name == "ApproxEq") {
                    // TODO
                    return "NotImplemented";
                }
                if (op_name == "And") {
                    return expression_table[parents[0]->id] + " && " +
                           expression_table[parents[1]->id];
                }
                if (op_name == "Or") {
                    return expression_table[parents[0]->id] + " || " +
                           expression_table[parents[1]->id];
                }
                if (op_name == "ZeroElem") {
                    return "af::iszero(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "IsNaN") {
                    return "af::isNaN(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "IsInf") {
                    return "af::isInf(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Select") {
                    return "af::select(" + expression_table[args[0]->id] + ", " +
                           expression_table[parents[0]->id] + ", " +
                           expression_table[parents[1]->id] + ")";
                }
                // Elementwise operators
                if (op_name == "Square") {
                    return expression_table[parents[0]->id] + " * " +
                           expression_table[parents[0]->id];
                }
                if (op_name == "Exp") {
                    return "af::exp(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Log") {
                    return "af::log(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Abs") {
                    return "af::abs(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Log1p") {
                    return "af::log1p(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Sin") {
                    return "af::sin(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Cos") {
                    return "af::cos(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Tan") {
                    return "af::tan(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Sinh") {
                    return "af::sinh(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Cosh") {
                    return "af::cosh(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Tanh") {
                    return "af::tanh(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Pow") {
                    // TODO
                    return "UnImplemented";
                }
                // Linear Algebra operators
                if (op_name == "Transpose") {
                    return "af::transpose(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "MatrixMul") {
                    if (parents.size() > 2) {
                        // TODO
                        return "Matmul implemented only for 2 parents";
                    }
                    // Have to check for transpose to use flags
                    std::string p0;
                    std::string flag0 = "AF_MAT_NONE";
                    std::string p1;
                    std::string flag1 = "AF_MAT_NONE";
                    std::string expr;
                    if (parents[0]->op->name == "Transpose") {
                        p0 = expression_table[parents[0]->op->get_parents()[0]->id];
                        flag0 = "AF_MAT_TRANS";
                    } else {
                        p0 = expression_table[parents[0]->id];
                    }
                    if (parents[1]->op->name == "Transpose") {
                        p1 = expression_table[parents[1]->op->get_parents()[0]->id];
                        flag1 = "AF_MAT_TRANS";
                    } else {
                        p1 = expression_table[parents[1]->id];
                    }
                    return "af::matmul(" + p0 + ", " + p1 + ", " + flag0 + ", " + flag1 + ")";
                }
                if (op_name == "MatrixInv") {
                    return "af::inverse(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Det") {
                    return "af::det(" + expression_table[parents[0]->id] + ")";
                }
                if (op_name == "Logdet") {
                    return "af::log(af::det(" + expression_table[parents[0]->id] + "))";
                }
                if (op_name == "Trace") {
                    return "af::sum(af::diag(" + expression_table[parents[0]->id] + "))";
                }
                // Shape operators
                if (op_name == "Diag") {
                    return "af::diag(" + expression_table[parents[0]->id] + ", 0, " +
                           std::to_string(node_in->shape[1] == 1) + ")";
                }
                if (op_name == "Reshape") {
                    std::string expression = "af::moddims(" + expression_table[parents[0]->id] + ", ";
                    for (int i = 0; i < 4; i++) {
                        expression += node_in->shape[i].to_string_with_star();
                        if (i < 3) {
                            expression += ", ";
                        }
                    }
                    return expression + ")";
                }
                if (op_name == "Reorder") {
                    std::string expression = "af::reorder(" + expression_table[parents[0]->id] + ", ";
                    auto order = dynamic_cast<op::Reorder *>(node_in->op.get())->order;
                    for (int i = 0; i < 4; i++) {
                        expression += order[i];
                        if (i < 3) {
                            expression += ", ";
                        }
                    }
                    return expression + ")";
                }
                // Indexing operators
                if (op_name == "Slice") {
                    // TODO
                    return "UnImplemented";
                }
                if (op_name == "SliceGrad") {
                    // TODO
                    return "UnImplemented";
                }
                if (op_name == "Index") {
                    // TODO
                    return "UnImplemented";
                }
                if (op_name == "IndexGrad") {
                    // TODO
                    return "UnImplemented";
                }
                // Multy-node operators
                if (op_name == "MaxAndArgMax") {
                    // TODO
                    return "UnImplemented";
                }
                if (op_name == "SortAndArgSort") {
                    // TODO
                    return "UnImplemented";
                }
                // Optimized operators
                if (op_name == "BinCrossEntropyLogit") {
                    std::string p = expression_table[parents[0]->id];
                    std::string sfx = expression_table[args[0]->id];
                    std::string sfmx = expression_table[args[1]->id];
                    return p + " * " + sfmx + " + (1.0 - " + p + ") * " + sfx;
                }

                if(op_name == "Cast"){
                    logger()->info() << parents[0]->op->name << " " << expression_table[parents[0]->id];
                    return expression_table[parents[0]->id];
                }

                return "Unreachable";
            }


            void write_af_interface(std::ofstream &f){
                f << "namespace metadiff{\n"
                        "    namespace shared{\n"
                        "        /** A shared variable is a like a static variable, which is synchronized between devices */\n"
                        "        class ArrayFireVariable: public SharedVariable {\n"
                        "        public:\n"
                        "            af::array value;\n"
                        "            ArrayFireVariable(size_t id,\n"
                        "                              af::array value,\n"
                        "                              std::string name):\n"
                        "                    SharedVariable(id, std::array<long long, 4> {value.dims(0), value.dims(1),\n"
                        "                                                                 value.dims(2), value.dims(3)},\n"
                        "                                   name),\n"
                        "                    value(value) {};\n"
                        "\n"
                        "            /** Converts an af_dtype to dType\n"
                        "             * TODO: Make proper exception when given complex type */\n"
                        "            static core::dType convert_af_dtype(af_dtype dtype){\n"
                        "                switch (dtype){\n"
                        "                    case af_dtype::b8 : return core::b8;\n"
                        "                    case af_dtype::u8 : return core::u8;\n"
                        "                    case af_dtype::u16: return core::u16;\n"
                        "                    case af_dtype::u32: return core::u32;\n"
                        "                    case af_dtype::u64: return core::u64;\n"
                        "                    case af_dtype::s16: return core::i16;\n"
                        "                    case af_dtype::s32: return core::i32;\n"
                        "                    case af_dtype::s64: return core::i64;\n"
                        "                    case af_dtype::f32 : return core::f32;\n"
                        "                    case af_dtype::f64: return core::f64;\n"
                        "                    default: throw 20;\n"
                        "                }\n"
                        "            }\n"
                        "\n"
                        "            core::dType get_dtype() const{\n"
                        "                return ArrayFireVariable::convert_af_dtype(value.type());\n"
                        "            }\n"
                        "        };\n"
                        "        \n"
                        "        typedef std::shared_ptr<ArrayFireVariable> AfShared;\n"
                        "\n"
                        "        static SharedPtr make_shared(af::array value, std::string name){\n"
                        "            SharedPtr ptr = std::make_shared<ArrayFireVariable>(shared_vars.size(), value, name);\n"
                        "            shared_vars.push_back(ptr);\n"
                        "            return ptr;\n"
                        "        }\n"
                        "    }\n"
                        "}\n"
                        "\n"
                        "using metadiff::shared::ArrayFireVariable;\n"
                        "using metadiff::shared::AfShared;\n"
                        "template <size_t T>\n\n"
                        "inline  AfShared get(std::vector<SharedPtr>& shared_vars){\n"
                        "     return std::static_pointer_cast<ArrayFireVariable>(shared_vars[T]);\n"
                        "}\n";
            }

//            template <typename size_t T>
//            inline  shared::AfShared get(std::vector<SharedPtr>& shared_vars){
//                return std::static_pointer_cast<shared::ArrayFireVariable>(shared_vars[T]);
//            }

            std::string shared_value(size_t index){
                return "get<" + std::to_string(index) + ">(shared_vars)->value";
//                return "std::static_pointer_cast<ArrayFireVariable>(shared_vars[" + std::to_string(index) + "])";
//                return "get_shared(" + std::to_string(index) + ", shared_vars)->value";
//                return "shared_vars[" + std::to_string(index) + "]->value";
            }
        };
#endif
    }
}

#endif //AUTODIFF_BACKENDS_ARRAYFIRE_H
