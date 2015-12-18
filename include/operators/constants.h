//
// Created by alex on 17/12/15.
//

#ifndef AUTODIFF_CONSTANTS_H
#define AUTODIFF_CONSTANTS_H
namespace metadiff{
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
#endif //AUTODIFF_CONSTANTS_H
