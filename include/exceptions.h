//
// Created by alex on 01/02/16.
//

#ifndef METADIFF_EXCEPTIONS_H
#define METADIFF_EXCEPTIONS_H

#include "iomanip"
namespace metadiff{
    namespace exceptions {
        using namespace core;

        void print_node(std::stringstream & stream, Node node){
            stream << "Id:"
            << std::setw(5) << node->id
            << " | Name: " << node->name
            << " | Shape: ("
            << std::setw(5) << node->shape[0] << ","
            << std::setw(5) << node->shape[1] << ","
            << std::setw(5) << node->shape[2] << ","
            << std::setw(5) << node->shape[3] << ")"
            << " | dtype: " << node->dtype
            << " | node_type: " << node->node_type
            << " | Op name: " << node->op->name << "\n";
        }

        class GraphError: public std::exception {
        public:
            NodeVec nodes;
            std::string msg;
            std::string nodes_description(){
                std::stringstream msg;
                msg << "Nodes:\n";
                for(size_t i=0;i<nodes.size(); i++){
                    print_node(msg, nodes[i]);
                }
                return msg.str();
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
            NodeVec targets;
            NodeVec inputs;
            Node missing;
            std::string msg;
            MissingRequiredInput(NodeVec targets, NodeVec inputs, Node missing) :
                    targets(targets),
                    inputs(inputs),
                    missing(missing) {
                std::stringstream msg;
                msg << "Error: Missing required input when trying to compile a function.\n"
                 << "Missing node:\n";
                print_node(msg, missing);
                msg << "Target nodes:\n";
                for(size_t i=0;i<targets.size();i++){
                    print_node(msg, targets[i]);
                }
                msg << "Provided inputs:\n";
                for(size_t i=0;i<inputs.size();i++){
                    print_node(msg, inputs[i]);
                }
                this->msg = msg.str();
            }

            const char *what() const throw() {
                return msg.c_str();
            }
        };

        class CompilationFailed : public std::exception {
        public:
            std::string msg;

            CompilationFailed(std::string msg) :
                    msg("Compilation failed due to:" + msg) { };

            const char *what() const throw() {
                return msg.c_str();
            }
        };
    }
}

#endif //METADIFF_EXCEPTIONS_H
