//
// Created by alex on 15/12/15.
//

#ifndef AUTODIFF_ELEMENTWISE_FUNC_H
#define AUTODIFF_ELEMENTWISE_FUNC_H
namespace autodiff {

    class Exp: public ElementwiseUnary {
    public:
        Exp(GraphInPtr graph, NodeInPtr parent) :
                ElementwiseUnary("Exp", graph, parent)
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if(messages.find(current) == messages.end()){
                return;
            }

            // Get the gradient with respect to this node
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            // Check for any surprises
            auto parent = this->parent.lock();
            if(parent->is_constant()) {
                throw_grad_type_error();
            }

            // Node computes f = exp(p)
            // => dE/dp = dE * f
            auto this_node = graph->nodes[current];
            auto op = std::make_shared<Mul>(graph, my_grad, this_node);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    class Log: public ElementwiseUnary {
    public:
        Log(GraphInPtr graph, NodeInPtr parent) :
                ElementwiseUnary("Log", graph, parent)
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if(messages.find(current) == messages.end()){
                return;
            }

            // Get the gradient with respect to this node
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            // Check for any surprises
            auto parent = this->parent.lock();
            if(parent->is_constant()) {
                throw_grad_type_error();
            }

            // Node computes f = log(p)
            // => dE/dp = dE * p^(-1)
            std::shared_ptr<Operator> op = std::make_shared<Neg>(graph, parent);
            auto parent_inv = graph->derived_node(op).lock();
            op = std::make_shared<Mul>(graph, my_grad, parent_inv);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    class Pow : public ElementwiseBinary {
    public:
        Pow(GraphInPtr graph, NodeInPtr parent1, NodeInPtr parent2) :
                ElementwiseBinary("Pow", graph, parent1, parent2) { };

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if (messages.find(current) == messages.end()) {
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            if (my_grad->name == "Derived Node" or my_grad->name == "") {
                my_grad->name = "Grad of " + std::to_string(current);
            } else {
                my_grad->name += "|Grad of " + std::to_string(current);
            }

            // Check for any surprises
            auto parent1 = this->parent1.lock();
            auto parent2 = this->parent2.lock();
            if (parent1->is_constant() and parent2->is_constant()) {
                 throw_grad_type_error();
            }

            // Node computes f = p_1^p_2
            // => dE/dp_1 = dE * p_2 * f / p_1
            // => dE/dp_2 = dE * f * log(p_1)
            auto this_node = graph->nodes[current];
            std::shared_ptr<Operator> op = std::make_shared<Mul>(graph, my_grad, this_node);
            auto this_node_times_grad = graph->derived_node(op);
            if(not parent1->is_constant()){
                op = std::make_shared<Div>(graph, parent1);
                auto parent1_inv = graph->derived_node(op);
                op = std::make_shared<Mul>(graph, this_node_times_grad, parent1_inv);
                auto parent_grad = graph->derived_node(op).lock();
                parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent1->id);
                send_grad_message(graph, parent1->id, parent_grad->id, messages);
            }
            if(not parent2->is_constant()){
                op = std::make_shared<Log>(graph, parent1);
                auto log_parent1 = graph->derived_node(op);
                op = std::make_shared<Mul>(graph, this_node_times_grad, log_parent1);
                auto parent_grad = graph->derived_node(op).lock();
                parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent1->id);
                send_grad_message(graph, parent2->id, parent_grad->id, messages);
            }
        }
    };
}

#endif //AUTODIFF_ELEMENTWISE_FUNC_H