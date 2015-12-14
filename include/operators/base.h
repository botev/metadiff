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
        Broadcast(std::weak_ptr<GraphInternal> graph,
                  NodeInPtr parent,
                  Shape to_shape):
                Operator(graph, "Broadcast"),
                parent(parent),
                to_shape(to_shape){
            auto parent = parent.lock();
            for(int i=0;i<4;i++){
                if(parent->shape[i] != 1 and parent->shape[i] != to_shape[i]){
                    throw IncompatibleShapes(name, {parent->id}, {parent->shape, to_shape});
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

        Sum(std::weak_ptr<GraphInternal> graph,
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
                        std::weak_ptr<GraphInternal> graph,
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
                          std::weak_ptr<GraphInternal> graph,
                          NodeInPtr p1,
                          NodeInPtr p2) :
                ElementwiseNary(name, graph, {p1, p2})
        {}
    };

    class ElementwiseUnary : public Operator{
    public:
        NodeInPtr parent;
        ElementwiseUnary(std::string const name,
                         std::weak_ptr<GraphInternal> graph,
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


    class ElementwiseAddition : public ElementwiseNary {
    public:
        ElementwiseAddition(std::weak_ptr<GraphInternal> graph, NodeInVec parents) :
                ElementwiseNary("Add", graph, parents)
        {}

        ElementwiseAddition(std::weak_ptr<GraphInternal> graph, NodeInPtr p1, NodeInPtr p2) :
                ElementwiseNary("Add", graph, {p1, p2})
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
                        auto op = std::make_shared<ElementwiseAddition>(graph, prev_msg, my_grad);
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
        auto op = std::make_shared<ElementwiseAddition>(graph, nodes_in);
        return Node(graph, graph->derived_node(op).lock()->id);
    };

    Node add(Node node1, Node node2){
        return add({node1, node2});
    };

    Node operator+(Node node1, Node node2){
        return add({node1, node2});
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
        auto op = std::make_shared<Sum>(this->graph, my_grad, this->get_axes());
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);

        // Add it to the already existing message to the parent on make this the first
        if (messages.find(parent->id) != messages.end()) {
            auto prev_msg = graph->nodes[messages[parent->id]];
            auto op = std::make_shared<ElementwiseAddition>(graph, prev_msg, parent_grad);
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
        auto op = std::make_shared<Broadcast>(this->graph, my_grad, parent->shape);
        auto parent_grad = graph->derived_node(op).lock();
        parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);

        // Add it to the already existing message to the parent on make this the first
        if (messages.find(parent->id) != messages.end()) {
            auto prev_msg = graph->nodes[messages[parent->id]];
            auto op = std::make_shared<ElementwiseAddition>(graph, prev_msg, parent_grad);
            messages[parent->id] = graph->derived_node(op).lock()->id;
        } else {
            messages[parent->id] = parent_grad->id;
        }
    }

    class Neg : public ElementwiseUnary {
    public:
        Neg(std::weak_ptr<GraphInternal> graph, NodeInPtr parent) :
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
            auto op = std::make_shared<Neg>(graph, my_grad);
            auto parent_grad = graph->derived_node(op).lock();
            parent_grad->name = "Grad msg " + std::to_string(current) + " -> " + std::to_string(parent->id);

            // Add it to the already existing message to the parent on make this the first
            if (messages.find(parent->id) != messages.end()) {
                auto prev_msg = graph->nodes[messages[parent->id]];
                auto op = std::make_shared<ElementwiseAddition>(graph, prev_msg, parent_grad);
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

    class Mul : public Operator {
    public:
        NodeInPtr p1;
        NodeInPtr p2;

        Mul(std::weak_ptr<GraphInternal> graph, NodeInPtr p1, NodeInPtr p2) :
                Operator(graph, "MUL"),
                p1(p1), p2(p2) { }

        NodeInVec get_parents() {
            return NodeInVec {p1, p2};
        }

        NodeInVec get_arguments() {
            return NodeInVec {};
        }

        ad_value_type get_value_type(){
            return ad_value_type::FLOAT ;
        }

        unsigned short get_gradient_level(){
            auto p1 = this->p1.lock();
            auto p2 = this->p1.lock();
            if(p1->grad_level > p2->grad_level){
                return p1->grad_level;
            } else{
                return p2->grad_level;
            }
        }

        std::array<SymInt,4> get_shape(){
            auto graph = this->graph.lock();
            auto p1 = this->p1.lock();
            auto p2 = this->p2.lock();
            if(p1->is_scalar()){
                return p2->shape;
            } else if(p2->is_scalar()){
                return p1->shape;
            } else if(p1->shape == p2->shape) {
                return p1->shape;
            } else{
                std::array<SymInt,4> shape;
                std::vector<size_t> dims;
                bool err = false;
                for(int i=0; i<4; i++){
                    if(p1->shape[i] != p2->shape[i]){
                        if(p1->shape[i] == 1){
                            dims.push_back(i);
                            shape[i] = p2->shape[i];
                        } else if(p2->shape[i] == 1){
                            dims.push_back(i);
                            shape[i] = p1->shape[i];
                        } else {
                            err = true;
                        }
                    } else {
                        shape[i] = p1->shape[i];
                    }
                }
                if(err){
                    throw IncompatibleShapes(this->name, {p1, p2});
                } else if(dims.size() == 0){
                    return shape;
                } else if(graph->broadcast == ad_implicit_broadcast::RAISE){
                    throw ImplicitBroadcast(this->name, this->get_parents());
                } else if(graph->broadcast == ad_implicit_broadcast::WARN){
                    auto msg = ImplicitBroadcast(this->name, this->get_parents());
                    std::cout << "WARNING:" << msg.get_message() << std::endl;
                    return shape;
                } else {
                    return shape;
                }
            }
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();
            if(messages.find(current) == messages.end()){
                return;
            }
            auto gradient = graph->nodes[messages[current]];
            auto p1 = this->p1.lock();
            auto p2 = this->p2.lock();
            if(p1->type != ad_node_type::CONSTANT
               and p1->type != ad_node_type::SYMBOLIC_INTEGER
               and p1->type != ad_node_type::CONSTANT_DERIVED) {
                auto op = std::make_shared<Mul>(graph, gradient, p2);
                auto p1_message = graph->derived_node(op).lock();
                p1_message->name = "Grad Msg [" + std::to_string(current) + "->" + std::to_string(p1->id) + "]";
                p1_message->grad_level = gradient->grad_level;
                if (messages.find(p1->id) != messages.end()) {
                    auto message = graph->nodes[messages[p1->id]];
                    auto add_op = std::make_shared<ElementwiseAddition>(graph, p1_message, message);
                    auto new_message = graph->derived_node(add_op).lock();
                    new_message->grad_level = gradient->grad_level;
                    messages[p1->id] = new_message->id;
                    new_message->name = "Grad of " + std::to_string(p1->id);
                    new_message->grad_level = gradient->grad_level;
                } else {
                    messages[p1->id] = p1_message->id;
                }
            }
            if(p2->type != ad_node_type::CONSTANT
               and p2->type != ad_node_type::SYMBOLIC_INTEGER
               and p2->type != ad_node_type::CONSTANT_DERIVED) {
                auto op = std::make_shared<Mul>(graph, gradient, p1);
                auto p2_message = graph->derived_node(op).lock();
                p2_message->name = "Grad Msg [" + std::to_string(current) + "->" + std::to_string(p2->id) + "]";
                p2_message->grad_level = gradient->grad_level;
                if (messages.find(p2->id) != messages.end()) {
                    auto message = graph->nodes[messages[p2->id]];
                    auto add_op = std::make_shared<ElementwiseAddition>(graph, p2_message, message);
                    auto new_message = graph->derived_node(add_op).lock();
                    messages[p2->id] = new_message->id;
                    new_message->name = "Grad of " + std::to_string(p1->id);
                    new_message->grad_level = gradient->grad_level;
                } else {
                    messages[p2->id] = p2_message->id;
                }
            }
        };
    };

    Node mul(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<Mul>(node1.graph, arg1, arg2);
        return Node(node1.graph, graph->derived_node(op).lock()->id);
    }

    Node operator*(Node node1, Node node2){
        return mul(node1, node2);
    };

}

#endif //AUTODIFF_BASE_H
