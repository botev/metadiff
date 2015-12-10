//
// Created by alex on 19/11/15.
//


#ifndef AUTODIFF_OPERATORS_H
#define AUTODIFF_OPERATORS_H

#include "core.h"

namespace autodiff {
    class Add : public Operator {
    public:
        std::weak_ptr<Node> p1;
        std::weak_ptr<Node> p2;

        Add(Graph* graph, std::weak_ptr<Node> p1, std::weak_ptr<Node> p2) :
                Operator(graph, "ADD"),
                p1(p1), p2(p2) { }

        std::vector<std::weak_ptr<Node>> get_parents() {
            return std::vector<std::weak_ptr<Node>> {p1, p2};
        }

        std::vector<std::weak_ptr<Node>> get_arguments() {
            return std::vector<std::weak_ptr<Node>> {};
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

        void generate_gradients(NodeId current, std::unordered_map<NodeId, NodeId> &messages) {
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
        std::weak_ptr<Node> p1;

        Neg(Graph* graph, std::weak_ptr<Node> p1) :
                Operator(graph, "NEG"),
                p1(p1) { }

        std::vector<std::weak_ptr<Node>> get_parents() {
            return std::vector<std::weak_ptr<Node>> {p1};
        }

        std::vector<std::weak_ptr<Node>> get_arguments() {
            return std::vector<std::weak_ptr<Node>> {};
        }

        ad_value_type get_value_type(){
            return ad_value_type::FLOAT ;
        }

        unsigned short get_gradient_level(){
            return p1.lock()->grad_level;
        }

        void generate_gradients(NodeId current, std::unordered_map<NodeId, NodeId> &messages) {
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
        std::weak_ptr<Node> p1;
        std::weak_ptr<Node> p2;

        Mul(Graph* graph, std::weak_ptr<Node> p1, std::weak_ptr<Node> p2) :
                Operator(graph, "MUL"),
                p1(p1), p2(p2) { }

        std::vector<std::weak_ptr<Node>> get_parents() {
            return std::vector<std::weak_ptr<Node>> {p1, p2};
        }

        std::vector<std::weak_ptr<Node>> get_arguments() {
            return std::vector<std::weak_ptr<Node>> {};
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

        void generate_gradients(NodeId current, std::unordered_map<NodeId, NodeId> &messages) {
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

    NodeId Graph::add(NodeId arg1_id, NodeId arg2_id) {
        auto arg1 = this->nodes[arg1_id];
        auto arg2 = this->nodes[arg2_id];
        NodeId result;
        if (arg1->type == ad_node_type::CONSTANT and arg2->type == ad_node_type::CONSTANT) {
            result = this->constant_node(arg1->value + arg2->value);
        }
        else {
            auto op = std::make_shared<Add>(this, arg1, arg2);
            result = this->derived_node(op);
        }
        return result;
    };

    NodeId Graph::neg(NodeId arg1_id) {
        auto arg1 = this->nodes[arg1_id];
        NodeId result;
        if (arg1->type == ad_node_type::CONSTANT) {
            result = this->constant_node(- arg1->value);
        }
        else {
            auto op = std::make_shared<Neg>(this, arg1);
            result = this->derived_node(op);
        }
        return result;
    };

    NodeId Graph::mul(NodeId arg1_id, NodeId arg2_id) {
        auto arg1 = this->nodes[arg1_id];
        auto arg2 = this->nodes[arg2_id];
        NodeId result;
        if (arg1->type == ad_node_type::CONSTANT and arg2->type == ad_node_type::CONSTANT) {
            result = this->constant_node(arg1->value * arg2->value);
        }
        else {
            auto op = std::make_shared<Mul>(this, arg1, arg2);
            result = this->derived_node(op);
        }
        return result;
    };
}

#endif //AUTODIFF_OPERATORS_H
