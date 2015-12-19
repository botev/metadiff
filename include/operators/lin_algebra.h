//
// Created by alex on 16/12/15.
//

#ifndef METADIFF_OPERATORS_LIN_ALGEBRA_H
#define METADIFF_OPERATORS_LIN_ALGEBRA_H

namespace metadiff{

    // Inverts the order of all non singular dimensions (numpy)
    class Transpose: public UnaryOperator{
    public:
        Transpose(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Transpose", graph, parent)
        {}

        Shape get_shape(){
            auto parent_shape = parent.lock()->shape;
            Shape shape {1, 1, 1, 1};
            int last_non_zero = 0;
            for(int i=3;i>=0;i--){
                if(parent_shape[i] != 1){
                    last_non_zero = i;
                    break;
                }
            }
            for(int i=0;i<=last_non_zero;i++){
                shape[i] = parent_shape[last_non_zero-i];
            }
            return shape;
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if (messages.find(current) == messages.end()) {
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            // Check for any surprises
            auto parent = this->parent.lock();
            if (parent->is_constant()) {
                throw_grad_type_error();
            }

            // Node computes f = p_1^T
            // => dE/dp_1 = dE/df^T
            std::shared_ptr<Operator> op = std::make_shared<Transpose>(graph, my_grad);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        }
    };

    Node Node::transpose() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Transpose>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node transpose(Node node){
        return node.transpose();
    }

    class MatrixMultiplication: public NaryOperator{
    public:
        MatrixMultiplication(GraphInPtr graph,
                             NodeInVec parents) :
                NaryOperator("MatrixMul", graph, parents)
        {
            if(not parents[0].lock()->is_matrix()){
                throw InvalidArguments(name, parents, "Parent 0 is not a matrix.");
            }
            for(int i=1;i<parents.size(); i++){
                if(not parents[i].lock()->is_matrix()){
                    throw InvalidArguments(name, parents, "Parent " + std::to_string(i) + " is not a matrix.");
                }
                if(parents[i-1].lock()->shape[1] != parents[i].lock()->shape[0]){
                    throw IncompatibleShapes(name, parents);
                }
            }
            shape = Shape{parents[0].lock()->shape[0], parents.back().lock()->shape[1], 1, 1};
        }

        MatrixMultiplication(GraphInPtr graph,
                             NodeInPtr parent1,
                             NodeInPtr parent2) :
        MatrixMultiplication(graph, {parent1, parent2})
        {}

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if (messages.find(current) == messages.end()) {
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            // Check for any surprises
            if(all_parent_const()){
                throw_grad_type_error();
            }

            // Node computes f = p_1 dot p_2 dot ... dot p_n
            // dE/dp_i = (p_1 dot ... dot p_{i-1})^T dE (p_{i+1} dot ... dot p_n)^T
            for(int i=0;i<parents.size();i++) {
                auto parent = parents[i].lock();
                if (not parent->is_constant()) {
                    std::vector<NodeInPtr> left_nodes;
                    std::vector<NodeInPtr> right_nodes;
                    for (int p = 0; p < i; p++) {
                        left_nodes.push_back(parents[p]);
                    }
                    for (int p = i + 1; p < parents.size(); p++) {
                        right_nodes.push_back(parents[p]);
                    }
                    NodeInPtr left_tr;
                    NodeInPtr right_tr;
                    if (left_nodes.size() == 1) {
                        auto op = std::make_shared<Transpose>(graph, left_nodes[0]);
                        left_tr = graph->derived_node(op, my_grad->grad_level);
                    } else if (left_nodes.size() > 1) {
                        std::shared_ptr<Operator> op = std::make_shared<MatrixMultiplication>(graph, left_nodes);
                        auto left_mul = graph->derived_node(op, my_grad->grad_level);
                        op = std::make_shared<Transpose>(graph, left_mul);
                        left_tr = graph->derived_node(op, my_grad->grad_level);
                    }
                    if (right_nodes.size() == 1) {
                        auto op = std::make_shared<Transpose>(graph, right_nodes[0]);
                        right_tr = graph->derived_node(op, my_grad->grad_level);
                    } else if (right_nodes.size() > 1) {
                        std::shared_ptr<Operator> op = std::make_shared<MatrixMultiplication>(graph, right_nodes);
                        auto right_mul = graph->derived_node(op, my_grad->grad_level);
                        op = std::make_shared<Transpose>(graph, right_mul);
                        right_tr = graph->derived_node(op, my_grad->grad_level);
                    }
                    std::shared_ptr<Operator> op;

                    if (not left_tr.lock()) {
                        op = std::make_shared<MatrixMultiplication>(graph, my_grad, right_tr);
                    } else if (not right_tr.lock()) {
                        op = std::make_shared<MatrixMultiplication>(graph, left_tr, my_grad);
                    } else {
                        op = std::make_shared<MatrixMultiplication>(graph, NodeInVec {left_tr, my_grad, right_tr});
                    }
                    auto parent_grad = graph->derived_node(op).lock();
                    parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
                    send_grad_message(graph, parent->id, parent_grad->id, messages);
                }
            }
        }
    };

    Node dot(std::vector<Node> nodes){
        auto graph = nodes[0].graph.lock();
        NodeInVec nodes_in;
        for(int i=0;i<nodes.size();i++){
            nodes_in.push_back(graph->nodes[nodes[i].id]);
        }
        auto op = std::make_shared<MatrixMultiplication>(graph, nodes_in);
        return Node(graph, graph->derived_node(op).lock()->id);
    };

    Node dot(Node node1, Node node2){
        return dot({node1, node2});
    }

    class MatrixInverse: public UnaryOperator{
    public:
        MatrixInverse(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("MatrixInv", graph, parent)
        {
            auto parent_shape = parent.lock()->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw UnknownError({parent}, "Operator 'MatrixInverse' takes only squared matrices");
            }
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if (messages.find(current) == messages.end()) {
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            // Check for any surprises
            auto parent = this->parent.lock();
            if (parent->is_constant()) {
                throw_grad_type_error();
            }

            // Node computes f = p_1^-1
            // => dE/dp_1 = - f^T dot dE dot f^T
            auto this_node = graph->nodes[current];
            std::shared_ptr<Operator> op = std::make_shared<Transpose>(graph, this_node);
            auto this_node_tr = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<MatrixMultiplication>(graph, NodeInVec {this_node_tr, my_grad, this_node_tr});
            auto minus_grad = graph->derived_node(op).lock();
            op = std::make_shared<Neg>(graph, minus_grad);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        }
    };

    Node Node::minv() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<MatrixInverse>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node minv(Node node){
        return node.minv();
    }

    class Determinant: public UnaryOperator{
    public:
        Determinant(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Det", graph, parent)
        {
            auto parent_shape = parent.lock()->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw UnknownError({parent}, "Operator 'Determinant' takes only squared matrices");
            }
            if(parent.lock()->v_type == BOOLEAN){
                throw UnknownError({parent}, "Operator 'Determinant' is not applicable for node of type BOOLEAN");
            }
        }

        Shape get_shape(){
            return {1, 1, 1, 1};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if (messages.find(current) == messages.end()) {
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            // Check for any surprises
            auto parent = this->parent.lock();
            if (parent->is_constant()) {
                throw_grad_type_error();
            }

            // Node computes f = det(p_1)
            // => dE/dp_1 = dE * f * p_1^(-1)^T
            auto this_node = graph->nodes[current];
            std::shared_ptr<Operator> op = std::make_shared<MatrixInverse>(graph, parent);
            auto parent_inv = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Transpose>(graph, parent_inv);
            auto parent_inv_tr = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Mul>(graph, NodeInVec {my_grad, this_node, parent_inv_tr});
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        }
    };

    Node Node::det() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Determinant>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node det(Node node){
        return node.det();
    }

    class LogDeterminant: public UnaryOperator{
    public:
        LogDeterminant(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("LogDet", graph, parent)
        {
            auto parent_shape = parent.lock()->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw UnknownError({parent}, "Operator 'Determinant' takes only squared matrices");
            }
            if(parent.lock()->v_type == BOOLEAN){
                throw UnknownError({parent}, "Operator 'Determinant' is not applicable for node of type BOOLEAN");
            }
        }

        Shape get_shape(){
            return {1, 1, 1, 1};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if (messages.find(current) == messages.end()) {
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            // Check for any surprises
            auto parent = this->parent.lock();
            if (parent->is_constant()) {
                throw_grad_type_error();
            }

            // Node computes f = logdet(p_1)
            // => dE/dp_1 = dE * p_1^(-1)^T
            std::shared_ptr<Operator> op = std::make_shared<MatrixInverse>(graph, parent);
            auto parent_inv = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Transpose>(graph, parent_inv);
            auto parent_inv_tr = graph->derived_node(op).lock();
            op = std::make_shared<Mul>(graph, my_grad, parent_inv_tr);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        }
    };

    Node Node::logdet() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<LogDeterminant>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node logdet(Node node){
        return node.logdet();
    }

    class Trace: public UnaryOperator{
    public:
        Trace(GraphInPtr graph, NodeInPtr parent) :
                UnaryOperator("Trace", graph, parent)
        {
            auto parent_shape = parent.lock()->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw UnknownError({parent}, "Operator 'Trace' takes only squared matrices");
            }
        }

        Shape get_shape(){
            return {1, 1, 1, 1};
        }

        ad_value_type get_value_type(){
            if(parent.lock()->v_type == BOOLEAN){
                return INTEGER;
            } else {
                return parent.lock()->v_type;
            }
        };

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if (messages.find(current) == messages.end()) {
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            // Check for any surprises
            auto parent = this->parent.lock();
            if (parent->is_constant()) {
                throw_grad_type_error();
            }

            // Node computes f = trace(p_1)
            // => dE/dp_1 = dE * eye
            std::shared_ptr<Operator> op = std::make_shared<Eye>(graph, parent->shape[0]);
            auto identity = graph->derived_node(op, my_grad->grad_level).lock();
            op = std::make_shared<Mul>(graph, my_grad, identity);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        }
    };

    Node Node::trace(){
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Trace>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node trace(Node node){
        return node.logdet();
    }
}
#endif //METADIFF_OPERATORS_LIN_ALGEBRA_H
