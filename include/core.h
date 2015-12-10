//
// Created by alex on 10/12/15.
//

#ifndef AUTODIFF_CORE_H
#define AUTODIFF_CORE_H

#include "vector"
#include "iostream"
#include <fstream>
#include "memory"
#include <unordered_map>
#include "arrayfire.h"

namespace autodiff {


    typedef size_t NodeId;

    enum ad_node_type{CONSTANT, INPUT, SHARED_INPUT, INPUT_DERIVED};
    enum ad_value_type{FLOAT, INTEGER, BOOLEAN};
    enum ad_float_type {f16, c16, f32, c32, f64, c64};
    enum ad_integer_type {s8, u8, s16, u16, s32, u32, s64, u64};
    enum ad_device_type {CPU, GPU};
    enum ad_implicit_broadcast {RAISE, WARN, QUIET};

    std::string to_string(ad_node_type const & type){
        switch(type){
            case ad_node_type::CONSTANT: return "CONSTANT";
            case ad_node_type::INPUT : return "INPUT";
            case ad_node_type::SHARED_INPUT : return "SHARED";
            case ad_node_type::INPUT_DERIVED: return "DERIVED";
        }
        return "UNEACHABLE";
    }

    std::string to_string(ad_value_type const & type){
        switch(type){
            case ad_value_type::FLOAT: return "FLOAT";
            case ad_value_type::INTEGER: return "INTEGER";
            case ad_value_type::BOOLEAN: return "BOOLEAN";
        }
        return "UNEACHABLE";
    }

    std::string to_string(ad_device_type const & type){
        switch(type){
            case ad_device_type::CPU: return "CPU";
            case ad_device_type::GPU: return "GPU";
        }
        return "UNEACHABLE";
    }


    std::ostream & operator<<(std::ostream & f, ad_node_type const & type) {
        f << to_string(type);
        return f;
    }

    std::ostream & operator<<(std::ostream & f, ad_value_type const & type) {
        f << to_string(type);
        return f;
    }

    std::ostream & operator<<(std::ostream & f, ad_device_type const & type) {
        f << to_string(type);
        return f;
    }

    class Device{
    public:
        ad_device_type type;
        unsigned int id;
        Device():
                type(ad_device_type::CPU),
                id(0)
        {};

        Device(const ad_device_type type, const unsigned int id):
                type(type),
                id(id)
        {};
    };

    std::string to_string(Device const & device){
        return autodiff::to_string(device.type) + "[" + std::to_string(device.id) + "]";
    }

    std::ostream & operator<<(std::ostream & f, Device const & device) {
        f << autodiff::to_string(device);
        return f;
    }


    class Graph;
    class Node;

    class Operator{
    public:
        Graph* graph;
        std::string name;
        Operator(Graph* graph, std::string name):
                name(name),
                graph(graph)
        {};
        virtual void generate_gradients(NodeId current, std::unordered_map<NodeId, NodeId>& messages) = 0;
        virtual ad_value_type get_value_type() = 0;
        virtual unsigned short get_gradient_level() = 0;
        virtual std::vector<std::weak_ptr<Node>> get_parents() = 0;
        virtual std::vector<std::weak_ptr<Node>> get_arguments() = 0;
        std::vector<std::weak_ptr<Node>> get_ancestors(){
            auto parents = this->get_parents();
            auto arguments = this->get_arguments();
            for(int i=0; i<arguments.size();i++){
                parents.push_back(arguments[i]);
            }
            return parents;
        }
    };

    class Node{
    public:
        Graph* graph;
        size_t id;
        std::string name;
        ad_node_type type;
        ad_value_type v_type;
        std::shared_ptr<Operator> op;
        std::vector<std::weak_ptr<Node>> children;
        unsigned short grad_level;
        Device device;
        af::array value;
        Node(Graph* graph, Device device):
                graph(graph),
                device(device)
        {}

        Node(Graph* graph, Device device, size_t id, std::string name,
             ad_node_type type, ad_value_type v_type, std::shared_ptr<Operator> op,
             unsigned short grad_level
        ):
                graph(graph),
                device(device),
                id(id),
                name(name),
                type(type),
                v_type(v_type),
                op(op),
                grad_level(grad_level)
        {}
    };

    class InputOperator : public Operator {
    public:
        InputOperator(Graph* graph): Operator(graph, "Input"){}

        ad_value_type get_value_type(){
            return ad_value_type::FLOAT ;
        }

        unsigned short get_gradient_level(){
            return 0;
        }

        std::vector<std::weak_ptr<Node>> get_parents(){
            return std::vector<std::weak_ptr<Node>> {};
        }

        std::vector<std::weak_ptr<Node>> get_arguments(){
            return std::vector<std::weak_ptr<Node>> {};
        }

        void generate_gradients(NodeId current, std::unordered_map<NodeId, NodeId>& messages){

        }
    };

    class Graph{
    public:
        std::vector<std::shared_ptr<Node>> nodes;
        std::string name;
        ad_float_type f_type;
        ad_integer_type i_type;
        Device default_device;

        Graph(){
            name = "Function";
            f_type = ad_float_type::f32;
            i_type = ad_integer_type::s32;
            // TODO Check if GPU is available and use that instead
            default_device = Device(CPU, 0);
        }

        NodeId input_node(){
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    "Input Node",
                    ad_node_type::INPUT ,
                    ad_value_type::FLOAT ,
                    std::make_shared<InputOperator>(this),
                    0
            );
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId constant_node(af::array value){
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT ,
                    ad_value_type::FLOAT ,
                    std::make_shared<InputOperator>(this),
                    0
            );
            result->value = value;
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId derived_node(std::shared_ptr<Operator> op) {
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    "Derived Node",
                    ad_node_type::INPUT_DERIVED,
                    op->get_value_type(),
                    op,
                    op->get_gradient_level()
            );
            this->nodes.push_back(result);
            auto ancestors = op->get_ancestors();
            for(int i=0;i<ancestors.size();i++){
                ancestors[i].lock()->children.push_back(result);
            }
            return result->id;
        }

        std::vector<NodeId> gradient(NodeId objective, std::vector<NodeId> params){
            std::unordered_map<NodeId, NodeId> grad_messages;
            auto target = this->nodes[objective];
            long n = this->nodes.size();
            auto unity_grad = this->constant_node(af::constant(1.0, af::dim4(10, 10, 1, 1)));
            this->nodes[unity_grad]->grad_level = target->grad_level + ((unsigned short) 1);
            grad_messages[target->id] = unity_grad;
            this->nodes[grad_messages[target->id]]->name = "Grad of " + std::to_string(objective);
//            std::cout << "Target " << objective << std::endl;
            int j=0;
            for(auto i=n;i>0;i--){
//                std::cout<<"Checking " << i << std::endl;
                if(grad_messages.find(i-1) != grad_messages.end()){
//                    std::cout << "Grad msg found" << std::endl;
//                    std::cout << this <<  "-" << this->nodes[i-1]->graph << std::endl;
                    this->nodes[i-1]->op->generate_gradients(i-1, grad_messages);
                }
            }
            std::vector<NodeId> grads;
            for(int i=0;i<params.size();i++){
                grads.push_back(grad_messages[params[i]]);
            }
            return grads;
        }
        NodeId add(NodeId arg1, NodeId arg2);
        NodeId neg(NodeId arg1);
        NodeId mul(NodeId arg1_id, NodeId arg2_id);
    };
}

#endif //AUTODIFF_CORE_H
