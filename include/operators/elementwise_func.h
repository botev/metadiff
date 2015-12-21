//
// Created by alex on 15/12/15.
//

#ifndef METADIFF_ELEMENTWISE_FUNC_H
#define METADIFF_ELEMENTWISE_FUNC_H
namespace metadiff {

    class ConstantConvert: public UnaryOperator {
        ConstantConvert(GraphInPtr graph, NodeInPtr parent) :
            UnaryOperator("Const", graph, parent)
        {};
    };

    class Exp: public UnaryOperator {
    public:
        Exp(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Exp", graph, parent)
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

    Node Node::exp() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Exp>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node exp(Node node){
        return node.exp();
    }

    class Log: public UnaryOperator {
    public:
        Log(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Log", graph, parent)
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
            auto parent_inv = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Mul>(graph, my_grad, parent_inv);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::log() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Log>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node log(Node node){
        return node.log();
    }

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
                auto parent1_inv = graph->derived_node(op, my_grad->grad_level);
                op = std::make_shared<Mul>(graph, this_node_times_grad, parent1_inv);
                auto parent_grad = graph->derived_node(op).lock();
                parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent1->id);
                send_grad_message(graph, parent1->id, parent_grad->id, messages);
            }
            if(not parent2->is_constant()){
                op = std::make_shared<Log>(graph, parent1);
                auto log_parent1 = graph->derived_node(op, my_grad->grad_level);
                op = std::make_shared<Mul>(graph, this_node_times_grad, log_parent1);
                auto parent_grad = graph->derived_node(op).lock();
                parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent1->id);
                send_grad_message(graph, parent2->id, parent_grad->id, messages);
            }
        }
    };

    Node pow(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<Pow>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node pow(Node node1, double value){
        auto node2 = af::constant(value, 1);
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[graph->constant_node(node2).id];
        auto op = std::make_shared<Pow>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node pow(double value, Node node2){
        auto node1 = af::constant(value, 1);
        auto graph = node2.graph.lock();
        auto arg1 = graph->nodes[graph->constant_node(node1).id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<Pow>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    class Abs: public UnaryOperator {
    public:
        Abs(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Abs", graph, parent)
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

            // Node computes f = abs(p)
            // => dE/dp = dE * (p>=0)
            auto zero = graph->nodes[graph->constant_node(af::constant(0.0, 1)).id];
            zero->grad_level = my_grad->grad_level;
            std::shared_ptr<Operator> op = std::make_shared<GreaterThanOrEqual>(graph, parent, zero);
            auto check = graph->derived_node(op).lock();
            op = std::make_shared<Mul>(graph, my_grad, check);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::abs() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Abs>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node abs(Node node){
        return node.abs();
    }

    class Sin: public UnaryOperator {
    public:
        Sin(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Sin", graph, parent)
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages);
    };

    Node Node::sin() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Sin>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node sin(Node node){
        return node.sin();
    }

    class Cos: public UnaryOperator {
    public:
        Cos(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Cos", graph, parent)
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages);
    };

    Node Node::cos(){
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Cos>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node cos(Node node){
        return node.cos();
    }
    void Sin::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
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

        // Node computes f = sin(p)
        // => dE/dp = dE * cos(p)
        std::shared_ptr<Operator> op = std::make_shared<Cos>(graph, parent);
        auto cos_parent = graph->derived_node(op, my_grad->grad_level).lock();
        op = std::make_shared<Mul>(graph, my_grad, cos_parent);
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
        send_grad_message(graph, parent->id, parent_grad->id, messages);
    };

    void Cos::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
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

        // Node computes f = cos(p)
        // => dE/dp = - dE * sin(p)
        std::shared_ptr<Operator> op = std::make_shared<Sin>(graph, parent);
        auto sin_parent = graph->derived_node(op, my_grad->grad_level).lock();
        op = std::make_shared<Mul>(graph, my_grad, sin_parent);
        auto minus_grad = graph->derived_node(op).lock();
        op = std::make_shared<Neg>(graph, minus_grad);
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
        send_grad_message(graph, parent->id, parent_grad->id, messages);
    };

    class Tan: public UnaryOperator {
    public:
        Tan(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Tan", graph, parent)
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

            // Node computes f = tan(p)
            // => dE/dp = dE / (cos(p)^2)
            auto this_node = graph->nodes[current];
            std::shared_ptr<Operator> op = std::make_shared<Cos>(graph, parent);
            auto cos_parent = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Square>(graph, cos_parent);
            auto cos_parent_sqr = graph->derived_node(op).lock();
            op = std::make_shared<Div>(graph, cos_parent_sqr);
            auto cos_parent_sqr_inv = graph->derived_node(op).lock();
            op = std::make_shared<Mul>(graph, my_grad, cos_parent_sqr_inv);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::tan() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Tan>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node tan(Node node){
        return node.tan();
    }

    class Cot: public UnaryOperator {
    public:
        Cot(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Cot", graph, parent)
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

            // Node computes f = cot(p)
            // => dE/dp = - dE / (sin(p)^2)
            auto this_node = graph->nodes[current];
            std::shared_ptr<Operator> op = std::make_shared<Sin>(graph, parent);
            auto sin_parent = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Square>(graph, sin_parent);
            auto sin_parent_sqr = graph->derived_node(op).lock();
            op = std::make_shared<Div>(graph, sin_parent_sqr);
            auto sin_parent_sqr_inv = graph->derived_node(op).lock();
            op = std::make_shared<Mul>(graph, my_grad, sin_parent_sqr_inv);
            auto minus_grad = graph->derived_node(op).lock();
            op = std::make_shared<Neg>(graph, minus_grad);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::cot() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Cot>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node cot(Node node){
        return node.cot();
    }

    class Sinh: public UnaryOperator {
    public:
        Sinh(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Sinh", graph, parent)
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages);
    };

    Node Node::sinh() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Sinh>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node sinh(Node node){
        return node.sinh();
    }

    class Cosh: public UnaryOperator {
    public:
        Cosh(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Cosh", graph, parent)
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages);
    };

    Node Node::cosh() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Cosh>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node cosh(Node node){
        return node.cosh();
    }

    void Sinh::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
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

        // Node computes f = sinh(p)
        // => dE/dp = dE * cosh(p)
        std::shared_ptr<Operator> op = std::make_shared<Cosh>(graph, parent);
        auto cosh_parent = graph->derived_node(op, my_grad->grad_level).lock();
        op = std::make_shared<Mul>(graph, my_grad, cosh_parent);
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
        send_grad_message(graph, parent->id, parent_grad->id, messages);
    };

    void Cosh::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
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

        // Node computes f = cosh(p)
        // => dE/dp = dE * sinh(p)
        std::shared_ptr<Operator> op = std::make_shared<Sinh>(graph, parent);
        auto sinh_parent = graph->derived_node(op, my_grad->grad_level).lock();
        op = std::make_shared<Mul>(graph, my_grad, sinh_parent);
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
        send_grad_message(graph, parent->id, parent_grad->id, messages);
    };


    class Tanh: public UnaryOperator {
    public:
        Tanh(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Tanh", graph, parent)
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

            // Node computes f = tanh(p)
            // => dE/dp = dE * (1 - f^2)
            auto this_node = graph->nodes[current];
            std::shared_ptr<Operator> op = std::make_shared<Square>(graph, this_node);
            auto this_node_sqr = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Neg>(graph, this_node_sqr);
            auto minus_this_node_sqr = graph->derived_node(op).lock();
            auto one = graph->nodes[graph->constant_node(af::constant(1.0, 1)).id];
            one->grad_level = my_grad->grad_level;
            op = std::make_shared<Add>(graph, minus_this_node_sqr, one);
            auto one_minus_this_node_sqr = graph->derived_node(op).lock();
            op = std::make_shared<Mul>(graph, my_grad, one_minus_this_node_sqr);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::tanh() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Tanh>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node tanh(Node node){
        return node.tanh();
    }

    class Coth: public UnaryOperator {
    public:
        Coth(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Coth", graph, parent)
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

            // Node computes f = coth(p)
            // => dE/dp = dE * (1 - f^2)
            auto this_node = graph->nodes[current];
            std::shared_ptr<Operator> op = std::make_shared<Square>(graph, this_node);
            auto this_node_sqr = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Neg>(graph, this_node_sqr);
            auto minus_this_node_sqr = graph->derived_node(op).lock();
            auto one = graph->nodes[graph->constant_node(af::constant(1.0, 1)).id];
            one->grad_level = my_grad->grad_level;
            op = std::make_shared<Add>(graph, minus_this_node_sqr, one);
            auto one_minus_this_node_sqr = graph->derived_node(op).lock();
            op = std::make_shared<Mul>(graph, my_grad, one_minus_this_node_sqr);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::coth() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Coth>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node coth(Node node){
        return node.coth();
    }

    class Sigmoid: public UnaryOperator {
    public:
        Sigmoid(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Sigmoid", graph, parent)
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

            // Node computes f = sigm(p)
            // => dE/dp = dE * f * (1 - f)
            auto this_node = graph->nodes[current];
            auto one = graph->nodes[graph->constant_node(af::constant(1.0, 1)).id];
            std::shared_ptr<Operator> op = std::make_shared<Neg>(graph, this_node);
            auto this_node_neg = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Add>(graph, one, this_node_neg);
            auto one_minus_this_node = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Mul>(graph, NodeInVec {my_grad, this_node, one_minus_this_node});
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::sigmoid() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Sigmoid>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node sigmoid(Node node){
        return node.sigmoid();
    }
}

#endif //METADIFF_ELEMENTWISE_FUNC_H
