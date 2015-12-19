//
// Created by alex on 17/12/15.
//

#ifndef METADIFF_OPERATORS_CONSTANTS_H
#define METADIFF_OPERATORS_CONSTANTS_H

namespace metadiff{
    class MakeConstant: public UnaryOperator{
    public:
        MakeConstant(GraphInPtr graph,
                     NodeInPtr parent):
                UnaryOperator("Const", graph, parent)
        {};

        ad_node_type get_node_type(){
            auto parent_type = parent.lock()->type;
            if(parent_type == CONSTANT){
                return CONSTANT;
            } else {
                return CONSTANT_DERIVED;
            }
        };

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();
            if (messages.find(current) != messages.end()) {
                this->throw_grad_type_error();
            }
            return;
        };
    };

    Node Node::constant() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<MakeConstant>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node to_constant(Node node){
        return node.constant();
    }

    class ConstantOperator: public Operator{
    public:
        Shape shape;
        ConstantOperator(std::string const name,
                         GraphInPtr graph):
                Operator(name, graph)
        {};

        NodeInVec get_parents() {
            return {};
        };

        ad_value_type get_value_type(){
            return FLOAT;
        };

        ad_node_type get_node_type(){
            return CONSTANT_DERIVED;
        };

        Shape get_shape(){
            return shape;
        }

        unsigned short get_gradient_level(){
            return 0;
        };

        NodeInVec get_arguments() {
            return NodeInVec {};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();
            if (messages.find(current) != messages.end()) {
                throw UnknownError({}, "The constant operator recieved a gradient message.");
            }
            return;
        };
    };

    class Eye: public ConstantOperator{
    public:
        Eye(GraphInPtr graph, SymInt size):
                ConstantOperator("Eye", graph)
        {
            shape = {size, size, 1, 1};
        }
    };
}
#endif //METADIFF_OPERATORS_CONSTANTS_H
