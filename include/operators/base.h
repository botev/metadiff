//
// Created by alex on 13/12/15.
//

#ifndef AUTODIFF_BASE_H
#define AUTODIFF_BASE_H

#include "autodiff.h"

namespace autodiff {

    class Broadcast : public Operator {
    public:
        NodeInPtr parent;
        Shape to_shape;
        Broadcast(GraphInPtr graph,
                  NodeInPtr parent,
                  Shape to_shape):
                Operator(graph, "Broadcast"),
                parent(parent),
                to_shape(to_shape){
            auto parent_shape = parent.lock()->shape;
            for(int i=0;i<4;i++){
                if(parent_shape[i] != 1 and parent_shape[i] != to_shape[i]){
                    throw IncompatibleShapes(name, {parent.lock()->id}, {parent_shape, to_shape});
                }
            }
        }

        ad_value_type get_value_type(){
            return parent.lock()->v_type;
        }

        unsigned short get_gradient_level(){
            return parent.lock()->grad_level;
        }

        Shape get_shape(){
            return to_shape;
        }

        NodeInVec get_parents(){
            return {parent};
        }

        NodeInVec get_arguments(){
            return {};
        }

        std::vector<size_t> get_axes(){
            std::vector<size_t> axes;
            auto p1_shape = this->parent.lock()->shape;
            for(int i=0;i<4;i++){
                if(p1_shape[i] != to_shape[i]){
                    axes.push_back(i);
                }
            }
            return axes;
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages);
    };

    Node Node::broadcast(Shape shape) {
        auto graph = this->graph.lock();
        auto op = std::make_shared<Broadcast>(this->graph, graph->nodes[this->id], shape);
        return Node(this->graph, graph->derived_node(op).lock()->id);
    }

    Node Node::broadcast_to(Node other) {
        return broadcast(graph.lock()->nodes[other.id]->shape);
    }

    class Sum : public Operator {
    public:
        NodeInPtr parent;
        std::vector<size_t> axes;

        Sum(GraphInPtr graph,
            NodeInPtr parent,
            std::vector<size_t> axes):
                Operator(graph, "Sum"),
                parent(parent),
                axes(axes)
        {
            if(axes.size() == 0){
                throw InvalidArguments(name, {parent}, "NULL");
            }
            bool err = false;
            if(axes.size() > 4){
                err = true;
            }
            bool checks[4] {false, false, false, false};
            for(int i=0;i<axes.size();i++){
                if(axes[i] > 3){
                    err = true;
                }
                if(checks[axes[i]]){
                    err = true;
                }
                checks[axes[i]] = true;
            }
            if(err){
                std::string axes_str;
                for(int i=0;i<axes.size();i++){
                    axes_str += std::to_string(axes[i]);
                    if(i < axes.size()-1){
                        axes_str += ", ";
                    }
                }
                throw InvalidArguments(name, {parent}, axes_str);
            }
        }

        ad_value_type get_value_type(){
            return parent.lock()->v_type;
        }

        unsigned short get_gradient_level(){
            return parent.lock()->grad_level;
        }

        Shape get_shape(){
            auto p_shape = parent.lock()->shape;
            for(int i=0;i<axes.size();i++){
                p_shape[axes[i]] = 1;
            }
            return p_shape;
        }

        NodeInVec get_parents(){
            return {parent};
        }

        NodeInVec get_arguments(){
            return {};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages);
    };

    Node Node::sum(std::vector<size_t> axes) {
        auto graph = this->graph.lock();
        auto op = std::make_shared<Sum>(this->graph, graph->nodes[this->id], axes);
        return Node(this->graph, graph->derived_node(op).lock()->id);
    }

    Node Node::sum() {
        return sum({0, 1, 2, 3});
    }

    class ElementwiseNary : public Operator{
    public:
        NodeInVec parents;
        Shape shape;
        ElementwiseNary(std::string const name,
                        GraphInPtr graph,
                        NodeInVec parents) :
                Operator(graph, name){
            try{
                shape = verify_shapes(parents);
            } catch(const int){
                throw IncompatibleShapes(name, parents);
            }
            for(int i=0;i<parents.size();i++){
                auto parent = parents[i].lock();
                if(parent->shape == shape or parent->is_scalar()){
                    this->parents.push_back(parents[i]);
                } else if(graph.lock()->broadcast == ad_implicit_broadcast::RAISE){
                    throw ImplicitBroadcast(name, parents);
                } else{
                    if(graph.lock()->broadcast == ad_implicit_broadcast::WARN){
                        auto msg = ImplicitBroadcast(name, parents);
                        std::cout << "WARNING:" << msg.get_message() << std::endl;
                    }
                    auto  op = std::make_shared<Broadcast>(this->graph, parents[i], shape);
                    this->parents.push_back(graph.lock()->derived_node(op));
                }
            }
        };

        NodeInVec get_parents() {
            return parents;
        };

        ad_value_type get_value_type(){
            auto top_type = BOOLEAN;
            for(int i=0;i<parents.size();i++){
                auto v_type = parents[i].lock()->v_type;
                if(v_type == FLOAT){
                    return FLOAT;
                }
                if(v_type == INTEGER){
                    top_type = INTEGER;
                }
            }
            return top_type;
        };

        unsigned short get_gradient_level(){
            unsigned short max_grad_level = 0;
            for(int i=0;i<parents.size();i++){
                auto grad_level = parents[i].lock()->grad_level;
                if(grad_level > max_grad_level){
                    max_grad_level = grad_level;
                }
            }
            return max_grad_level;
        };

        std::array<SymInt,4> get_shape(){
            return shape;
        }

        NodeInVec get_arguments() {
            return NodeInVec {};
        }
    };

    class ElementwiseBinary : public ElementwiseNary{
        ElementwiseBinary(std::string name,
                          GraphInPtr graph,
                          NodeInPtr parent1,
                          NodeInPtr parent2) :
                ElementwiseNary(name, graph, {parent1, parent2})
        {}
    };

    class ElementwiseUnary : public Operator{
    public:
        NodeInPtr parent;
        ElementwiseUnary(std::string const name,
                         GraphInPtr graph,
                         NodeInPtr parent):
                Operator(graph, name),
                parent(parent)
        {};

        NodeInVec get_parents() {
            return {parent};
        };

        ad_value_type get_value_type(){
            return parent.lock()->v_type;
        };

        unsigned short get_gradient_level(){
            return parent.lock()->grad_level;
        };

        std::array<SymInt,4> get_shape(){
            return parent.lock()->shape;
        }

        NodeInVec get_arguments() {
            return NodeInVec {};
        }
    };


    class Add : public ElementwiseNary {
    public:
        Add(GraphInPtr graph, NodeInVec parents) :
                ElementwiseNary("Add", graph, parents)
        {}

        Add(GraphInPtr graph, NodeInPtr parent1, NodeInPtr parent2) :
                ElementwiseNary("Add", graph, {parent1, parent2})
        {}

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if(messages.find(current) == messages.end()){
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            if(my_grad->name == "Derived Node" or my_grad->name == ""){
                my_grad->name = "Grad of " + std::to_string(current);
            } else {
                my_grad->name += "|Grad of " + std::to_string(current);
            }

            // Check for any surprises
            bool check = true;
            for(int i=0;i<parents.size();i++){
                // No need to generate gradient message as it is my_grad for addition
                auto parent = parents[i].lock();
                if(parent->type != ad_node_type::CONSTANT
                   and parent->type != ad_node_type::SYMBOLIC_INTEGER
                   and parent->type != ad_node_type::CONSTANT_DERIVED) {
                    // Add it to the already existing message to the parent on make this the first
                    if (messages.find(parent->id) != messages.end()) {
                        auto prev_msg = graph->nodes[messages[parent->id]];
                        auto op = std::make_shared<Add>(graph, prev_msg, my_grad);
                        messages[parent->id] = graph->derived_node(op).lock()->id;
                    } else {
                        messages[parent->id] = my_grad->id;
                    }
                    check = false;
                }
            }

            if(check){
                std::string type_str;
                for(int i=0;i<parents.size();i++){
                    type_str += to_string(parents[i].lock()->type);
                    if(i < parents.size() - 1){
                        type_str += ", ";
                    }
                }
                throw UnkownError(parents, "Gradient message present, but parents are " + type_str);
            }
        };
    };

    Node add(std::vector<Node> nodes){
        auto graph = nodes[0].graph.lock();
        NodeInVec nodes_in;
        for(int i=0;i<nodes.size();i++){
            nodes_in.push_back(graph->nodes[nodes[i].id]);
        }
        auto op = std::make_shared<Add>(graph, nodes_in);
        return Node(graph, graph->derived_node(op).lock()->id);
    };

    Node add(Node node1, Node node2){
        return add({node1, node2});
    };

    Node operator+(Node node1, Node node2){
        return add({node1, node2});
    };

    Node operator+(Node node){
        return node;
    };

    void Broadcast::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
        auto graph = this->graph.lock();

        // Check for any incoming messages
        if(messages.find(current) == messages.end()){
            return;
        }

        // Get the gradient with respect to this node, alter the name
        auto my_grad = graph->nodes[messages[current]];
        if(my_grad->name == "Derived Node" or my_grad->name == ""){
            my_grad->name = "Grad of " + std::to_string(current);
        } else {
            my_grad->name += "|Grad of " + std::to_string(current);
        }

        // Check for any surprises
        auto parent = this->parent.lock();
        if(parent->type == ad_node_type::CONSTANT
           and parent->type == ad_node_type::SYMBOLIC_INTEGER
           and parent->type == ad_node_type::CONSTANT_DERIVED) {
            throw UnkownError({parent},
                              "Gradient message present, but parent is " + to_string(parent->type));
        }

        // Generate the parent's gradient message
        // If this node computed any broadcast(p, shape), the gradient is sum(dE, axes)
        auto op = std::make_shared<Sum>(this->graph, my_grad, this->get_axes());
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);

        // Add it to the already existing message to the parent on make this the first
        if (messages.find(parent->id) != messages.end()) {
            auto prev_msg = graph->nodes[messages[parent->id]];
            auto op = std::make_shared<Add>(graph, prev_msg, parent_grad);
            messages[parent->id] = graph->derived_node(op).lock()->id;
        } else {
            messages[parent->id] = parent_grad->id;
        }
    }

    void Sum::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
        auto graph = this->graph.lock();

        // Check for any incoming messages
        if(messages.find(current) == messages.end()){
            return;
        }

        // Get the gradient with respect to this node, alter the name
        auto my_grad = graph->nodes[messages[current]];
        if(my_grad->name == "Derived Node" or my_grad->name == ""){
            my_grad->name = "Grad of " + std::to_string(current);
        } else {
            my_grad->name += "|Grad of " + std::to_string(current);
        }

        // Check for any surprises
        auto parent = this->parent.lock();
        if(parent->type == ad_node_type::CONSTANT
           and parent->type == ad_node_type::SYMBOLIC_INTEGER
           and parent->type == ad_node_type::CONSTANT_DERIVED) {
            throw UnkownError({parent},
                              "Gradient message present, but parent is " + to_string(parent->type));
        }

        // Generate the parent's gradient message
        // If this node computed any sum(p), the gradient is broadcast(dE, p.shape)
        auto op = std::make_shared<Broadcast>(this->graph, my_grad, parent->shape);
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);

        // Add it to the already existing message to the parent on make this the first
        if (messages.find(parent->id) != messages.end()) {
            auto prev_msg = graph->nodes[messages[parent->id]];
            auto op = std::make_shared<Add>(graph, prev_msg, parent_grad);
            messages[parent->id] = graph->derived_node(op).lock()->id;
        } else {
            messages[parent->id] = parent_grad->id;
        }
    }

    class Neg : public ElementwiseUnary {
    public:
        Neg(GraphInPtr graph, NodeInPtr parent) :
                ElementwiseUnary("Neg", graph, parent)
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if(messages.find(current) == messages.end()){
                return;
            }

            // Get the gradient with respect to this node, alter the name
            auto my_grad = graph->nodes[messages[current]];
            if(my_grad->name == "Derived Node" or my_grad->name == ""){
                my_grad->name = "Grad of " + std::to_string(current);
            } else {
                my_grad->name += "|Grad of " + std::to_string(current);
            }

            // Check for any surprises
            auto parent = this->parent.lock();
            if(parent->type == ad_node_type::CONSTANT
               and parent->type == ad_node_type::SYMBOLIC_INTEGER
               and parent->type == ad_node_type::CONSTANT_DERIVED) {
                throw UnkownError({parent},
                                  "Gradient message present, but parent is " + to_string(parent->type));
            }

            // Generate the parent's gradient message
            // If this node computes -p the gradient is -dE
            auto op = std::make_shared<Neg>(graph, my_grad);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);

            // Add it to the already existing message to the parent on make this the first
            if (messages.find(parent->id) != messages.end()) {
                auto prev_msg = graph->nodes[messages[parent->id]];
                auto op = std::make_shared<Add>(graph, prev_msg, parent_grad);
                messages[parent->id] = graph->derived_node(op).lock()->id;
            } else {
                messages[parent->id] = parent_grad->id;
            }
        };
    };

    Node neg(Node node){
        auto graph = node.graph.lock();
        auto arg = graph->nodes[node.id];
        auto op = std::make_shared<Neg>(node.graph, arg);
        return Node(node.graph, graph->derived_node(op).lock()->id);
    }

    Node operator-(Node node){
        return neg(node);
    }

    Node operator-(Node node1, Node node2){
        return add({node1, neg(node2)});
    }

    class Mul : public ElementwiseNary {
    public:
        Mul(GraphInPtr graph, NodeInVec parents) :
                ElementwiseNary("Mul", graph, parents)
        {};

        Mul(GraphInPtr graph, NodeInPtr p1, NodeInPtr p2) :
                ElementwiseNary("Mul", graph, {p1, p2})
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages);
    };

    Node mul(std::vector<Node> nodes){
        auto graph = nodes[0].graph.lock();
        NodeInVec nodes_in;
        for(int i=0;i<nodes.size();i++){
            nodes_in.push_back(graph->nodes[nodes[i].id]);
        }
        auto op = std::make_shared<Mul>(graph, nodes_in);
        return Node(graph, graph->derived_node(op).lock()->id);
    };

    Node mul(Node node1, Node node2){
        return mul({node1, node2});
    }

    Node operator*(Node node1, Node node2){
        return mul({node1, node2});
    };

    class Div : public ElementwiseUnary {
    public:
        Div(GraphInPtr graph, NodeInPtr parent) :
                ElementwiseUnary("Div", graph, parent)
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages);
    };

    Node div(Node node1, Node node2){
        auto graph = node2.graph.lock();
        auto op = std::make_shared<Div>(node2.graph, graph->nodes[node2.id]);
        auto node2_div = graph->derived_node(op);
        return mul({node1, Node(graph, node2_div.lock()->id)});
    }

    Node operator/(Node node1, Node node2){
        return div(node1, node2);
    };

    class Square : public ElementwiseUnary {
    public:
        Square(GraphInPtr graph, NodeInPtr parent) :
        ElementwiseUnary("Square", graph, parent)
        {};

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages);
    };

    Node square(Node node){
        auto graph = node.graph.lock();
        auto op = std::make_shared<Square>(node.graph, graph->nodes[node.id]);
        return Node(node.graph, graph->derived_node(op).lock()->id);
    }

    void Mul::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
        auto graph = this->graph.lock();

        // Check for any incoming messages
        if(messages.find(current) == messages.end()){
            return;
        }

        // Get the gradient with respect to this node, alter the name
        auto my_grad = graph->nodes[messages[current]];
        if(my_grad->name == "Derived Node" or my_grad->name == ""){
            my_grad->name = "Grad of " + std::to_string(current);
        } else {
            my_grad->name += "|Grad of " + std::to_string(current);
        }

        // Check for any surprises
        bool check = true;
        if(parents.size() == 2){
            // Special case when only two parents
            for(int i=0;i<2;i++){
                auto parent = parents[i].lock();
                auto other_parent = parents[1-i].lock();
                if(parent->type != ad_node_type::CONSTANT
                   and parent->type != ad_node_type::SYMBOLIC_INTEGER
                   and parent->type != ad_node_type::CONSTANT_DERIVED) {
                    // Generate the parent's gradient message
                    // If my node computed p1 * p2
                    // Gradient msg to p1 is p2 * dE and to p2 is p1 * dE
                    auto op = std::make_shared<Mul>(graph, my_grad, other_parent);
                    auto parent_grad = graph->derived_node(op);
                    if (messages.find(parent->id) != messages.end()) {
                        auto prev_msg = graph->nodes[messages[parent->id]];
                        auto op = std::make_shared<Add>(graph, prev_msg, my_grad);
                        messages[parent->id] = graph->derived_node(op).lock()->id;
                    } else {
                        messages[parent->id] = my_grad->id;
                    }
                    check = false;
                }
            }
        } else {
            auto this_node = graph->nodes[current];
            for(int i=0;i<parents.size();i++){
                auto parent = parents[i].lock();
                if(parent->type != ad_node_type::CONSTANT
                   and parent->type != ad_node_type::SYMBOLIC_INTEGER
                   and parent->type != ad_node_type::CONSTANT_DERIVED) {
                    // Generate the parent's gradient message
                    // If my node computed p_1 * p_2 * p-3 ... * p_n = P
                    // Gradient msg to p_i is P * dE / p_i
                    std::shared_ptr<Operator> op = std::make_shared<Div>(graph, parent);
                    auto parent_inv = graph->derived_node(op);
                    op = std::make_shared<Mul>(graph, NodeInVec{my_grad, this_node, parent_inv});
                    auto parent_grad = graph->derived_node(op);
                    if (messages.find(parent->id) != messages.end()) {
                        auto prev_msg = graph->nodes[messages[parent->id]];
                        auto op = std::make_shared<Add>(graph, prev_msg, my_grad);
                        messages[parent->id] = graph->derived_node(op).lock()->id;
                    } else {
                        messages[parent->id] = my_grad->id;
                    }
                    check = false;
                }
            }
        }
        if(check){
            std::string type_str;
            for(int i=0;i<parents.size();i++){
                type_str += to_string(parents[i].lock()->type);
                if(i < parents.size() - 1){
                    type_str += ", ";
                }
            }
            throw UnkownError(parents, "Gradient message present, but parents are " + type_str);
        }
    }

    void Div::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
        auto graph = this->graph.lock();

        // Check for any incoming messages
        if(messages.find(current) == messages.end()){
            return;
        }

        // Get the gradient with respect to this node, alter the name
        auto my_grad = graph->nodes[messages[current]];
        if(my_grad->name == "Derived Node" or my_grad->name == ""){
            my_grad->name = "Grad of " + std::to_string(current);
        } else {
            my_grad->name += "|Grad of " + std::to_string(current);
        }

        // Check for any surprises
        auto parent = this->parent.lock();
        if(parent->type == ad_node_type::CONSTANT
           and parent->type == ad_node_type::SYMBOLIC_INTEGER
           and parent->type == ad_node_type::CONSTANT_DERIVED) {
            throw UnkownError({parent},
                              "Gradient message present, but parent is " + to_string(parent->type));
        }

        // Generate the parent's gradient message
        // If my node computed p^-1
        // Gradient msg to p is (-((p^2)^-1)) * dE
        auto parent_node = graph->nodes[parent->id];
        std::shared_ptr<Operator> op = std::make_shared<Square>(graph, parent);
        auto parent_sqr = graph->derived_node(op);
        op = std::make_shared<Div>(graph, parent_sqr);
        auto parent_sqr_inv = graph->derived_node(op);
        op = std::make_shared<Neg>(graph, parent_sqr_inv);
        auto minus_parent_sqr_inv = graph->derived_node(op);
        op = std::make_shared<Mul>(graph, my_grad, minus_parent_sqr_inv);
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);

        // Add it to the already existing message to the parent on make this the first
        if (messages.find(parent->id) != messages.end()) {
            auto prev_msg = graph->nodes[messages[parent->id]];
            auto op = std::make_shared<Add>(graph, prev_msg, parent_grad);
            messages[parent->id] = graph->derived_node(op).lock()->id;
        } else {
            messages[parent->id] = parent_grad->id;
        }
    }

    void Square::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
        auto graph = this->graph.lock();

        // Check for any incoming messages
        if(messages.find(current) == messages.end()){
            return;
        }

        // Get the gradient with respect to this node, alter the name
        auto my_grad = graph->nodes[messages[current]];
        if(my_grad->name == "Derived Node" or my_grad->name == ""){
            my_grad->name = "Grad of " + std::to_string(current);
        } else {
            my_grad->name += "|Grad of " + std::to_string(current);
        }

        // Check for any surprises
        auto parent = this->parent.lock();
        if(parent->type == ad_node_type::CONSTANT
           and parent->type == ad_node_type::SYMBOLIC_INTEGER
           and parent->type == ad_node_type::CONSTANT_DERIVED) {
            throw UnkownError({parent},
                              "Gradient message present, but parent is " + to_string(parent->type));
        }

        // Generate the parent's gradient message
        // If my node computed p^2
        // Gradient msg to p is 2 * p * dE
        auto parent_node = graph->nodes[parent->id];
        auto two = graph->nodes[graph->constant_node(2.0).id];
        auto op = std::make_shared<Mul>(graph, NodeInVec {my_grad, two, parent});
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);

        // Add it to the already existing message to the parent on make this the first
        if (messages.find(parent->id) != messages.end()) {
            auto prev_msg = graph->nodes[messages[parent->id]];
            auto op = std::make_shared<Add>(graph, prev_msg, parent_grad);
            messages[parent->id] = graph->derived_node(op).lock()->id;
        } else {
            messages[parent->id] = parent_grad->id;
        }
    }

}

#endif //AUTODIFF_BASE_H
