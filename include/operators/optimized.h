//
// Created by alex on 19/12/15.
//

#ifndef METADIFF_OPERATORS_OPTIMIZED_H
#define METADIFF_OPERATORS_OPTIMIZED_H

namespace metadiff {
    class Softplus : public UnaryOperator{
    public:
        double threshold;
        Softplus(GraphInPtr graph,
                 NodeInPtr parent,
                double threshold = 50):
                UnaryOperator("Softplus", graph, parent),
                threshold(threshold)
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

            // Node computes f = softplus(p)
            // => dE/dp = dE * sigmoid(p)
            std::shared_ptr<Operator> op = std::make_shared<Sigmoid>(graph, parent);
            auto sigmoid_p = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Mul>(graph, my_grad, sigmoid_p);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::softplus(double threshold) {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Softplus>(graph, arg, threshold);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node softplus(Node node, double threshold = 50){
        return node.softplus(threshold);
    }

    class BinaryCrossEntropyLogit : public ElementwiseNary {
    public:
        BinaryCrossEntropyLogit(GraphInPtr graph, NodeInVec parents):
                ElementwiseNary("BinCrossEntropyLogit", graph, parents)
        {
            if(parents.size() != 4){
                throw InvalidArguments(name, parents, "Operator 'BinCrossEntropyLogit' takes exactly 4 arguments.");
            }
        }

        ad_value_type get_value_type() {
            return FLOAT;
        };

        void generate_gradients(size_t current, std::unordered_map <size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if (messages.find(current) == messages.end()) {
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            // Check for any surprises
            if (all_parent_const()) {
                throw_grad_type_error();
            }

            // Parents - 1)p, 2) sf(x), 3) sf(-x), 4)x
            // Node computes f = - p * log(q) - (1-p) * log(1-q)
            // log(q) = -sf(-x), log(1-q) = -sf(x)
            // Node computes f = p * sf(-x) + (1 - p)*sf(x) = p*(sf(-x)-sf(x)) + sf(x)
            // dE/dp = dE * (sf(-x)-sf(x))
            // dE/dx = dE * (q-p) = dE * (sigmoid(x) - p)
            auto p = parents[0].lock();
            auto sf = parents[1].lock();
            auto sfm = parents[2].lock();
            auto x = parents[3].lock();
            if(not p->is_constant()){
                std::shared_ptr<Operator> op = std::make_shared<Neg>(graph, sf);
                auto minus_sf = graph->derived_node(op, my_grad->grad_level).lock();
                op = std::make_shared<Add>(graph, sfm, minus_sf);
                auto sfm_minus_sf = graph->derived_node(op, my_grad->grad_level).lock();
                op = std::make_shared<Mul>(graph, my_grad, sfm_minus_sf);
                auto parent_grad = graph->derived_node(op, my_grad->grad_level).lock();
                parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(p->id);
                send_grad_message(graph, p->id, parent_grad->id, messages);
            }
            if(not x->is_constant()){
                std::shared_ptr<Operator> op = std::make_shared<Sigmoid>(graph, x);
                auto q = graph->derived_node(op, my_grad->grad_level).lock();
                op = std::make_shared<Neg>(graph, p);
                auto minus_p = graph->derived_node(op, my_grad->grad_level).lock();
                op = std::make_shared<Add>(graph, q, minus_p);
                auto p_minus_q = graph->derived_node(op, my_grad->grad_level).lock();
                op = std::make_shared<Mul>(graph, my_grad, p_minus_q);
                auto parent_grad = graph->derived_node(op, my_grad->grad_level).lock();
                parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(x->id);
                send_grad_message(graph, x->id, parent_grad->id, messages);
            }
        }
    };

    Node binary_cross_entropy_logit(Node p, Node x){
        auto graph = p.graph.lock();
        auto sf = softplus(x);
        auto sfm = softplus(-x);
        auto op = std::make_shared<BinaryCrossEntropyLogit>(graph,
                                                            NodeInVec{
                                                                    graph->nodes[p.id],
                                                                    graph->nodes[sf.id],
                                                                    graph->nodes[sfm.id],
                                                                    graph->nodes[x.id] });
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node relu(Node x){
        auto graph = x.graph.lock();
        Node ch = x.abs();
        return graph->constant_node(af::constant(0.5, 1)) * (x + ch);
    }
};


#endif //METADIFF_OPERATORS_OPTIMIZED_H
