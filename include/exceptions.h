//
// Created by alex on 01/02/16.
//

#ifndef METADIFF_EXCEPTIONS_H
#define METADIFF_EXCEPTIONS_H

#include "iomanip"
namespace metadiff{
    namespace exceptions {
        using namespace core;

        class GraphError: public std::exception {
        public:
            NodeVec nodes;
            std::string msg;
            std::string nodes_description(){
                std::stringstream msg;
                msg << "Nodes:\n";
                for(size_t i=0;i<nodes.size(); i++){
                    msg << "Id:"
                    << std::setw(5) << nodes[i]->id
                    << " | Shape: ("
                    << std::setw(5) << nodes[i]->shape[0] << ","
                    << std::setw(5) << nodes[i]->shape[1] << ","
                    << std::setw(5) << nodes[i]->shape[2] << ","
                    << std::setw(5) << nodes[i]->shape[3] << ")"
                    << " | dtype: " << nodes[i]->dtype
                    << " | node_type: " << nodes[i]->node_type
                    << " | Op name: " << nodes[i]->op->name;
                }
                return msg.str();
            }

            void set_message(std::string msg){

            }
            GraphError(NodeVec nodes):
                    nodes(nodes){};

            const char *what() const throw() {
                return msg.c_str();
            }
        };

        class UnsupportedGradient : public GraphError {
        public:
            UnsupportedGradient(Node node):
                    GraphError(NodeVec{node}) {
                this->msg = "\nError: Taking gradient is only possible with respect to scalar objectives.\n" +
                            nodes_description() + "\n";
            }
        };

        class WrongGradient : public GraphError {
        public:
            WrongGradient(NodeVec inputs, std::string op_name):
                    GraphError(inputs) {
                this->msg = "\nError: The gradient node with id " + std::to_string(inputs[1]->id) +
                            " was sent to node with id " + std::to_string(inputs[0]->id) +
                            " and operator " + inputs[0]->op->name + " , but all its parents are constant.\n" +
                            nodes_description() + "\n";
            }
        };

        class OtherError: public GraphError{
        public:
            OtherError(NodeVec inputs, std::string msg):
            GraphError(inputs) {
                    this->msg = "\nError: " + msg + "\n" +
                                nodes_description() + "\n";
            }
        };

        class OperatorError : public GraphError{
        public:
            std::string op_name;
            std::string err;
            OperatorError(NodeVec inputs, std::string op_name, std::string err):
                    GraphError(inputs), op_name(op_name), err(err) {
                this->msg = "\nError in operator " + op_name + "\n" +
                            "Description: " + err + "\n" +
                            nodes_description() + "\n";
            }
        };

        class ImplicitBroadcast : public OperatorError {
        public:
            ImplicitBroadcast(NodeVec inputs, std::string op_name) :
                    OperatorError(inputs, op_name, "Performing implicit broadcast.") {}
        };

        class IncompatibleShapes : public OperatorError {
        public:
            IncompatibleShapes(NodeVec inputs, std::string op_name):
                    OperatorError(inputs, op_name, "Incompatible shapes of inputs") {}
        };

        class InvalidArguments : public OperatorError {
        public:
            InvalidArguments(NodeVec inputs, std::string op_name, std::string err):
                    OperatorError(inputs, op_name, err) {}
        };

        class MissingRequiredInput : public std::exception {
        public:
            std::vector<size_t> target_ids;
            size_t input_id;

            MissingRequiredInput(std::vector<size_t> targets, size_t input_id) :
                    target_ids(target_ids),
                    input_id(input_id) { }

            MissingRequiredInput(NodeVec targets, size_t input_id) :
                    input_id(input_id) {
                for (int i = 0; i < targets.size(); i++) {
                    this->target_ids.push_back(targets[i].unwrap()->id);
                }
            }

            const char *what() const throw() {
                std::string msg = "Missing required input " + std::to_string(input_id) + " for targets [";
                for (int i = 0; i < target_ids.size(); i++) {
                    msg += std::to_string(target_ids[i]);
                    if (i < target_ids.size() - 1) {
                        msg += ", ";
                    }
                }
                msg += "]";
                return msg.c_str();
            }
        };

        class CompilationFailed : public std::exception {
        public:
            std::string msg;

            CompilationFailed(std::string msg) :
                    msg("Compilation failed due to:\n" + msg) { };

            const char *what() const throw() {
                return msg.c_str();
            }
        };
    }
}

#endif //METADIFF_EXCEPTIONS_H
