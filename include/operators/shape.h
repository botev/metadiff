//
// Created by alex on 18/12/15.
//

#ifndef METADIFF_OPERATORS_SHAPE_H
#define METADIFF_OPERATORS_SHAPE_H

namespace metadiff{

    class Diagonal: public UnaryOperator{
    public:
        Shape shape;
        Diagonal(GraphInPtr graph, NodeInPtr parent):
                UnaryOperator("Diag", graph, parent){
            auto parent_node = this->parent.lock();
            if(not parent_node->is_matrix()){
                throw InvalidArguments(name, {parent}, "Parent is not a vector or a sqyare matrix.");
            }
            if(parent_node->is_vector()){
                shape = {parent_node->shape[0], parent_node->shape[0], 1, 1};
            } else if(parent_node->shape[0] != parent_node->shape[1]){
                throw InvalidArguments(name, {parent}, "Parent is not a vector or a sqyare matrix.");
            } else {
                shape = {parent_node->shape[0], 1, 1, 1};
            }
        };

        Shape get_shape(){
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

            // Node computes f = diag(p_1)
            // => dE/dp_1 = diag(dE)
            std::shared_ptr<Operator> op = std::make_shared<Diagonal>(graph, my_grad);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        }
    };

    Node Node::diag() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Diagonal>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node diag(Node node){
        return node.transpose();
    }

    class Reshape: public UnaryOperator{
    public:
        Shape shape;
        Reshape(GraphInPtr graph, NodeInPtr parent, Shape shape):
                UnaryOperator("Reshape", graph, parent),
                shape(shape){
            auto parent_node = this->parent.lock();
            auto product_parent = number_of_elements(parent_node->shape);
            auto product_shape = number_of_elements(this->shape);
            if(product_parent != product_shape){
                std::string shape_str;
                throw InvalidArguments(name, {parent_node->id}, {parent_node->shape, this->shape},
                                       "Operator 'Reshape' must not change the total number of elemetns");
            }
        };

        Shape get_shape(){
            return shape;
        }

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

            // Node computes f = reshape(p, s)
            // => dE/dp = reshape(dE, p.shape)
            std::shared_ptr<Operator> op = std::make_shared<Reshape>(graph, my_grad, parent->shape);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::reshape(Shape shape){
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Reshape>(graph, arg, shape);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node reshape(Node node, Shape shape){
        return node.reshape(shape);
    }

    Node Node::flatten(size_t ndim) {
        if(ndim == 0 or ndim > 4){
            auto op = graph.lock()->nodes[id]->op;
            throw InvalidArguments(op->name, op->get_parents(), "Flatten accepts only values in the range [1,4]");
        }
        Shape parent_shape = graph.lock()->nodes[id]->shape;
        Shape shape = parent_shape;
        for(int i=3;i>=ndim;i--){
            shape[i-1] = shape[i] * shape[i-1];
            shape[i] = 1;
        }
        return reshape(shape);
    }

    Node flatten(Node node, size_t ndim = 1){
        return node.flatten(ndim);
    }

    class Reorder: public UnaryOperator{
    public:
        std::array<size_t ,4> order;
        Reorder(GraphInPtr graph, NodeInPtr parent, std::array<size_t, 4> order):
                UnaryOperator("Reorder", graph, parent),
                order(order){
            bool check[4] {false, false, false, false};
            for(int i=0;i<4;i++){
                if(order[i] > 4){
                    throw InvalidArguments(name, {this->parent},
                                           "The ordering for 'Reorder' must contain elements in the range [0,3]");
                }
                if(check[order[i]]){
                    throw InvalidArguments(name, {this->parent},
                                           "The ordering for 'Reorder' must not have repeating elements");
                }
            }
        };

        Shape get_shape(){
            auto parent = this->parent.lock();
            return {parent->shape[order[0]], parent->shape[order[1]],
                    parent->shape[order[2]], parent->shape[order[3]]};
        }

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

            // Node computes f = reshape(p, s)
            // => dE/dp = reshape(dE, p.shape)
            std::shared_ptr<Operator> op = std::make_shared<Reshape>(graph, my_grad, parent->shape);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        };
    };

    Node Node::reorder(std::array<size_t, 4> order){
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<Reorder>(graph, arg, order);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node reorder(Node node, std::array<size_t, 4> order){
        return node.reorder(order);
    }

    Node Node::reorder(size_t dim1, size_t dim2, size_t dim3, size_t dim4){
        return reorder({dim1, dim2, dim3, dim4});
    }

    Node reorder(Node node, size_t dim1, size_t dim2, size_t dim3=2, size_t dim4=3){
        return node.reorder({dim1, dim2, dim3, dim4});
    }
}
#endif //METADIFF_OPERATORS_SHAPE_H
