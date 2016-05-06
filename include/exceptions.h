//
// Created by alex on 01/02/16.
//

#ifndef METADIFF_EXCEPTIONS_H
#define METADIFF_EXCEPTIONS_H

namespace metadiff{
    namespace exceptions {
        using namespace core;

        /** Format is Id | node_type | Shape | dtype | Op name */
        void print_node(std::stringstream & stream, Node node){
            stream << "|" << std::setw(4) << node->id << " | " << node->node_type << "| "
            << node->shape << " | " << node->dtype << " | "
            << node->op->name;
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
                    if(i < nodes.size() - 1){
                        msg << std::endl;
                    }
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
                this->msg = "Error: Taking gradient is only possible with respect to scalar objectives. " +
                            nodes_description();
            }
        };

        class WrongGradient : public GraphError {
        public:
            WrongGradient(NodeVec inputs, std::string op_name):
                    GraphError(inputs) {
                this->msg = "Error: The gradient node with id " + std::to_string(inputs[1]->id) +
                            " was sent to node with id " + std::to_string(inputs[0]->id) +
                            " and operator " + inputs[0]->op->name + " , but all its parents are constant. " +
                            nodes_description();
            }
        };

        class OtherError: public GraphError{
        public:
            OtherError(NodeVec inputs, std::string msg):
            GraphError(inputs) {
                    this->msg = "Error: " + msg + " " + nodes_description();
            }
        };

        class OperatorError : public GraphError{
        public:
            std::string op_name;
            std::string err;
            OperatorError(NodeVec inputs, std::string op_name, std::string err):
                    GraphError(inputs), op_name(op_name), err(err) {
                this->msg = err + " " + nodes_description();
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
                msg << "\nTarget nodes:\n";
                for(size_t i=0;i<targets.size();i++){
                    print_node(msg, targets[i]);
                    msg << std::endl;
                }
                msg << "Provided inputs:\n";
                for(size_t i=0;i<inputs.size();i++){
                    print_node(msg, inputs[i]);
                    if(i < inputs.size() - 1){
                        msg << std::endl;
                    }
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

    namespace core{
        /** Given an error executes the errorPolicy */
        void operate_policy(errorPolicy policy,
                            std::shared_ptr<spdlog::logger> const logger,
                            exceptions::GraphError const & err);
    }
}

#endif //METADIFF_EXCEPTIONS_H
