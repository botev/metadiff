//
// Created by alex on 13/12/15.
//

#ifndef AUTODIFF_BASE_H
#define AUTODIFF_BASE_H

#include "autodiff.h"

namespace autodiff {

    class BroadcastOperator : public Operator {
    public:
        std::weak_ptr<NodeInternal> p1;
        Shape to_shape;
        BroadcastOperator(std::weak_ptr<GraphInternal> graph,
                          std::weak_ptr<NodeInternal> p1,
                          Shape to_shape):
                Operator(graph, "Broadcast"),
                p1(p1),
                to_shape(to_shape){
            auto parent = p1.lock();
            for(int i=0;i<4;i++){
                if(parent->shape[i] != 1 and parent->shape[i] != to_shape[i]){
                    // TODO proper exception
                    throw "wtv";
                }
            }
        }

        ad_value_type get_value_type(){
            return p1.lock()->v_type;
        }

        unsigned short get_gradient_level(){
            return p1.lock()->grad_level;
        }

        Shape get_shape(){
            return to_shape;
        }

        std::vector<std::weak_ptr<NodeInternal>> get_parents(){
            return {p1};
        }

        std::vector<std::weak_ptr<NodeInternal>> get_arguments(){
            return {};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t>& messages){
            //TODO sum operator
        };
    };

    class Add : public Operator {
    public:
        std::weak_ptr<NodeInternal> p1;
        std::weak_ptr<NodeInternal> p2;

        Add(std::weak_ptr<GraphInternal> graph, std::weak_ptr<NodeInternal> p1, std::weak_ptr<NodeInternal> p2) :
                Operator(graph, "ADD"),
                p1(p1), p2(p2) { }

        std::vector<std::weak_ptr<NodeInternal>> get_parents() {
            return std::vector<std::weak_ptr<NodeInternal>> {p1, p2};
        }

        std::vector<std::weak_ptr<NodeInternal>> get_arguments() {
            return std::vector<std::weak_ptr<NodeInternal>> {};
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
                    throw ImplicitBroadcast(this->name, this->get_parents(), dims);
                } else if(graph->broadcast == ad_implicit_broadcast::WARN){
                    auto msg = ImplicitBroadcast(this->name, this->get_parents(), dims);
                    std::cout << "WARNING:" << msg.get_message() << std::endl;
                    return shape;
                } else {
                    return shape;
                }
            }
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();
//            std::cout << "G" << std::endl;
//            std::cout << "G " << this->graph << std::endl;
//            std::cout<< "This: " << graph->nodes[current]->id << std::endl;
            if(messages.find(current) == messages.end()){
                return;
            }
            auto gradient = graph->nodes[messages[current]];
            auto p1 = this->p1.lock();
            auto p2 = this->p2.lock();
//            std::cout<< p1->id << ", " << p2->id << std::endl;
            if(p1->type != ad_node_type::CONSTANT) {
                if (messages.find(p1->id) != messages.end()) {
                    auto message = graph->nodes[messages[p1->id]];
                    auto op = std::make_shared<Add>(graph, gradient, message);
                    auto new_message = graph->derived_node(op);
                    graph->nodes[new_message]->grad_level = gradient->grad_level;
                    messages[p1->id] = new_message;
                    graph->nodes[new_message]->name = "Grad of " + std::to_string(p1->id);
                } else {
                    messages[p1->id] = gradient->id;
                }
            }
            if(p2->type != ad_node_type::CONSTANT) {
                if (messages.find(p2->id) != messages.end()) {
                    auto message = graph->nodes[messages[p2->id]];
                    auto op = std::make_shared<Add>(graph, gradient, message);
                    auto new_message = graph->derived_node(op);
                    graph->nodes[new_message]->grad_level = gradient->grad_level;
                    graph->nodes[new_message]->name = "Grad of " + std::to_string(p2->id);
                    messages[p2->id] = new_message;
                } else {
                    messages[p2->id] = gradient->id;
                }
            }
        };
    };

    class Neg : public Operator {
    public:
        std::weak_ptr<NodeInternal> p1;

        Neg(std::weak_ptr<GraphInternal> graph, std::weak_ptr<NodeInternal> p1) :
                Operator(graph, "NEG"),
                p1(p1) { }

        std::vector<std::weak_ptr<NodeInternal>> get_parents() {
            return std::vector<std::weak_ptr<NodeInternal>> {p1};
        }

        std::vector<std::weak_ptr<NodeInternal>> get_arguments() {
            return std::vector<std::weak_ptr<NodeInternal>> {};
        }

        ad_value_type get_value_type(){
            return ad_value_type::FLOAT ;
        }

        unsigned short get_gradient_level(){
            return p1.lock()->grad_level;
        }

        std::array<SymInt,4> get_shape(){
            return p1.lock()->shape;
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();
            if(messages.find(current) == messages.end()){
                return;
            }
            auto p1 = this->p1.lock();
            if(p1->type != ad_node_type::CONSTANT) {
                auto gradient = graph->nodes[messages[current]];
                auto op = std::make_shared<Neg>(graph, gradient);
                auto new_grad = graph->nodes[graph->derived_node(op)];
                new_grad->name = "Grad of " + std::to_string(p1->id);
                new_grad->grad_level = gradient->grad_level;
                if (messages.find(p1->id) != messages.end()) {
                    auto message = graph->nodes[messages[p1->id]];
                    auto op = std::make_shared<Add>(graph, new_grad, message);
                    auto new_message = graph->derived_node(op);
                    graph->nodes[new_message]->grad_level = new_grad->grad_level;
                    messages[p1->id] = new_message;
                } else {
                    messages[p1->id] = new_grad->id;
                }
            }
        };
    };

    class Mul : public Operator {
    public:
        std::weak_ptr<NodeInternal> p1;
        std::weak_ptr<NodeInternal> p2;

        Mul(std::weak_ptr<GraphInternal> graph, std::weak_ptr<NodeInternal> p1, std::weak_ptr<NodeInternal> p2) :
                Operator(graph, "MUL"),
                p1(p1), p2(p2) { }

        std::vector<std::weak_ptr<NodeInternal>> get_parents() {
            return std::vector<std::weak_ptr<NodeInternal>> {p1, p2};
        }

        std::vector<std::weak_ptr<NodeInternal>> get_arguments() {
            return std::vector<std::weak_ptr<NodeInternal>> {};
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
                    throw ImplicitBroadcast(this->name, this->get_parents(), dims);
                } else if(graph->broadcast == ad_implicit_broadcast::WARN){
                    auto msg = ImplicitBroadcast(this->name, this->get_parents(), dims);
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
            if(p1->type != ad_node_type::CONSTANT) {
                auto op = std::make_shared<Mul>(graph, gradient, p2);
                auto p1_message = graph->nodes[graph->derived_node(op)];
                p1_message->name = "Grad Msg [" + std::to_string(current) + "->" + std::to_string(p1->id) + "]";
                p1_message->grad_level = gradient->grad_level;
                if (messages.find(p1->id) != messages.end()) {
                    auto message = graph->nodes[messages[p1->id]];
                    auto add_op = std::make_shared<Add>(graph, p1_message, message);
                    auto new_message = graph->derived_node(add_op);
                    graph->nodes[new_message]->grad_level = gradient->grad_level;
                    messages[p1->id] = new_message;
                    graph->nodes[new_message]->name = "Grad of " + std::to_string(p1->id);
                    graph->nodes[new_message]->grad_level = gradient->grad_level;
                } else {
                    messages[p1->id] = p1_message->id;
                }
            }
            if(p2->type != ad_node_type::CONSTANT) {
                auto op = std::make_shared<Mul>(graph, gradient, p1);
                auto p2_message = graph->nodes[graph->derived_node(op)];
                p2_message->name = "Grad Msg [" + std::to_string(current) + "->" + std::to_string(p2->id) + "]";
                p2_message->grad_level = gradient->grad_level;
                if (messages.find(p2->id) != messages.end()) {
                    auto message = graph->nodes[messages[p2->id]];
                    auto add_op = std::make_shared<Add>(graph, p2_message, message);
                    auto new_message = graph->derived_node(add_op);
                    messages[p2->id] = new_message;
                    graph->nodes[new_message]->name = "Grad of " + std::to_string(p1->id);
                    graph->nodes[new_message]->grad_level = gradient->grad_level;
                } else {
                    messages[p2->id] = p2_message->id;
                }
            }
        };
    };

    Node add(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<Add>(node1.graph, arg1, arg2);
        return Node(node1.graph, graph->derived_node(op));
    };

    Node operator+(Node node1, Node node2){
        return add(node1, node2);
    };

    Node neg(Node node){
        auto graph = node.graph.lock();
        auto arg = graph->nodes[node.id];
        auto op = std::make_shared<Neg>(node.graph, arg);
        return Node(node.graph, graph->derived_node(op));
    }

    Node operator-(Node node){
        return neg(node);
    }

    Node mul(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<Mul>(node1.graph, arg1, arg2);
        return Node(node1.graph, graph->derived_node(op));
    }

    Node operator*(Node node1, Node node2){
        return mul(node1, node2);
    };

}

#endif //AUTODIFF_BASE_H
