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
//#include "arrayfire.h"
#include "symbolic.h"

namespace autodiff {
    const size_t N = 100;
    typedef size_t NodeId;
    typedef symbolic::SymbolicPolynomial<N, unsigned short> SymInt;

    enum ad_node_type{CONSTANT, INPUT, SHARED_INPUT, INPUT_DERIVED, SYMBOLIC_INTEGER};
    enum ad_value_type{FLOAT, INTEGER, BOOLEAN};
    enum ad_device_type {CPU, GPU};
    enum ad_implicit_broadcast {RAISE, WARN, QUIET};
    enum ad_float_type {f16, c16, f32, c32, f64, c64};
    enum ad_integer_type {s8, u8, s16, u16, s32, u32, s64, u64};

    std::string to_string(ad_node_type const & type){
        switch(type){
            case ad_node_type::CONSTANT: return "CONSTANT";
            case ad_node_type::INPUT : return "INPUT";
            case ad_node_type::SHARED_INPUT : return "SHARED";
            case ad_node_type::INPUT_DERIVED: return "DERIVED";
            case ad_node_type::SYMBOLIC_INTEGER: return "SYMBOLIC_INTEGER";
        }
        return "UNREACHABLE";
    }

    std::string to_string(ad_value_type const & type){
        switch(type){
            case ad_value_type::FLOAT: return "FLOAT";
            case ad_value_type::INTEGER: return "INTEGER";
            case ad_value_type::BOOLEAN: return "BOOLEAN";
        }
        return "UNREACHABLE";
    }

    std::string to_string(ad_device_type const & type){
        switch(type){
            case ad_device_type::CPU: return "CPU";
            case ad_device_type::GPU: return "GPU";
        }
        return "UNREACHABLE";
    }

    std::string to_string(ad_implicit_broadcast const & type){
        switch(type){
            case ad_implicit_broadcast::RAISE: return "Raise";
            case ad_implicit_broadcast::WARN: return "Warn";
            case ad_implicit_broadcast::QUIET: return "Quiet";
        }
        return "UNREACHABLE";
    }

    std::string to_string(ad_float_type const & type){
        switch(type){
            case ad_float_type::f16: return "f16";
            case ad_float_type::c16: return "c16";
            case ad_float_type::f32: return "f32";
            case ad_float_type::c32: return "c32";
            case ad_float_type::f64: return "f64";
            case ad_float_type::c64: return "c64";
        }
        return "UNREACHABLE";
    }

    std::string to_string(ad_integer_type const & type){
        switch(type){
            case ad_integer_type::s8: return "s8";
            case ad_integer_type::u8: return "u8";
            case ad_integer_type::s16: return "s16";
            case ad_integer_type::u16: return "u16";
            case ad_integer_type::s32: return "s32";
            case ad_integer_type::u32: return "u32";
            case ad_integer_type::s64: return "s64";
            case ad_integer_type::u64: return "u64";
        }
        return "UNREACHABLE";
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

    std::ostream & operator<<(std::ostream & f, ad_implicit_broadcast const & type) {
        f << to_string(type);
        return f;
    }

    std::ostream & operator<<(std::ostream & f, ad_float_type const & type) {
        f << to_string(type);
        return f;
    }

    std::ostream & operator<<(std::ostream & f, ad_integer_type const & type) {
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

    class SharedVariable{
    public:
        SharedVariable(){};
    };


    class Graph;
    class Node;

    class Operator{
    public:
        Graph* graph;
        std::string name;
        Operator(Graph* graph, std::string name):
                graph(graph),
                name(name)
        {};
        virtual void generate_gradients(NodeId current, std::unordered_map<NodeId, NodeId>& messages) = 0;
        virtual ad_value_type get_value_type() = 0;
        virtual unsigned short get_gradient_level() = 0;
        virtual std::array<SymInt,4> get_shape() = 0;
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
        Device device;
        size_t id;
        std::string name;
        ad_node_type type;
        ad_value_type v_type;
        std::array<SymInt,4> shape;
        std::shared_ptr<Operator> op;
        std::vector<std::weak_ptr<Node>> children;
        unsigned short grad_level;
        // Only for constant nodes
//        af::array value;
        double *f_value;
        double fs_value;
        int *i_value;
        int is_value;
        // Only for symbolic integers
        symbolic::SymbolicMonomial<N, unsigned short> integer_value;

        Node(Graph* graph, Device device):
                graph(graph),
                device(device)
        {}

        Node(Graph* graph, Device device, size_t id, std::string name,
             ad_node_type type, ad_value_type v_type, std::array<SymInt,4> shape,
             std::shared_ptr<Operator> op,
             unsigned short grad_level):
                graph(graph),
                device(device),
                id(id),
                name(name),
                type(type),
                v_type(v_type),
                op(op),
                grad_level(grad_level),
                shape(shape)
        {}

        bool is_scalar(){
            for(int i=0; i < 4; i++){
                if(this->shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_vector(){
            for(int i=1; i < 4; i++){
                if(this->shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_vector_strict(){
            for(int i=0; i < 1; i++){
                if(this->shape[i] == 1){
                    return false;
                }
            }
            for(int i=1; i < 4; i++){
                if(this->shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_matrix(){
            for(int i=2; i < 4; i++){
                if(this->shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_matrix_strict(){
            for(int i=0; i < 2; i++){
                if(this->shape[i] == 1){
                    return false;
                }
            }
            for(int i=2; i < 4; i++){
                if(this->shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_tensor3(){
            for(int i=3; i < 4; i++){
                if(this->shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_tensor3_strict(){
            for(int i=0; i < 3; i++){
                if(this->shape[i] == 1){
                    return false;
                }
            }
            for(int i=3; i < 4; i++){
                if(this->shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_tensor4_strict(){
            for(int i=0; i < 4; i++){
                if(this->shape[i] == 1){
                    return false;
                }
            }
            return true;
        }
    };

    class InputOperator : public Operator {
    public:
        InputOperator(Graph* graph): Operator(graph, "Input"){}

        ad_value_type get_value_type(){
            return ad_value_type::FLOAT;
        }

        unsigned short get_gradient_level(){
            return 0;
        }

        std::array<SymInt,4> get_shape(){
            return std::array<SymInt,4>{0, 0, 0, 0};
        }

        std::vector<std::weak_ptr<Node>> get_parents(){
            return std::vector<std::weak_ptr<Node>> {};
        }

        std::vector<std::weak_ptr<Node>> get_arguments(){
            return std::vector<std::weak_ptr<Node>> {};
        }

        void generate_gradients(NodeId current, std::unordered_map<NodeId, NodeId>& messages){};
    };

    class Graph{
    public:
        std::vector<std::shared_ptr<Node>> nodes;
        std::string name;
        Device default_device;
        ad_float_type f_type;
        ad_integer_type i_type;
        ad_implicit_broadcast broadcast;
        size_t sym_integer_count;

        Graph(){
            name = "Function";
            sym_integer_count = 0;
            // TODO Check if GPU is available and use that instead
            default_device = Device(CPU, 0);
            f_type = ad_float_type::f32;
            i_type = ad_integer_type::s32;
            broadcast = ad_implicit_broadcast::RAISE;
        }

        SymInt get_new_symbolic_integer(){
            this->sym_integer_count++;
            return SymInt(this->sym_integer_count-1);
        }

        NodeId constant_node(double* value, std::array<size_t, 4> dims){
            std::array<SymInt,4> shape {SymInt::as_polynomial(dims[0]),
                                        SymInt::as_polynomial(dims[1]),
                                        SymInt::as_polynomial(dims[2]),
                                        SymInt::as_polynomial(dims[3])};
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT ,
                    ad_value_type::FLOAT ,
                    shape,
                    std::make_shared<InputOperator>(this),
                    0
            );
            result->f_value = value;
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId constant_node(double value){
            std::array<SymInt,4> shape {SymInt::one(), SymInt::one(), SymInt::one(), SymInt::one()};
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT ,
                    ad_value_type::FLOAT ,
                    shape,
                    std::make_shared<InputOperator>(this),
                    0
            );
            result->fs_value = value;
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId constant_node(int* value, std::array<size_t, 4> dims){
            std::array<SymInt,4> shape {SymInt::as_polynomial(dims[0]),
                                        SymInt::as_polynomial(dims[1]),
                                        SymInt::as_polynomial(dims[2]),
                                        SymInt::as_polynomial(dims[3])};
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT ,
                    ad_value_type::INTEGER,
                    shape,
                    std::make_shared<InputOperator>(this),
                    0
            );
            result->i_value = value;
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId constant_node(int value){
            std::array<SymInt,4> shape {SymInt::one(), SymInt::one(), SymInt::one(), SymInt::one()};
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT ,
                    ad_value_type::INTEGER,
                    shape,
                    std::make_shared<InputOperator>(this),
                    0
            );
            result->is_value = value;
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
                    op->get_shape(),
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
            auto unity_grad = this->constant_node(1.0);
            this->nodes[unity_grad]->grad_level = target->grad_level + ((unsigned short) 1);
            grad_messages[target->id] = unity_grad;
            this->nodes[grad_messages[target->id]]->name = "Grad of " + std::to_string(objective);
            int j=0;
            for(size_t i=n;i>0;i--){
                if(grad_messages.find(i-1) != grad_messages.end()){
                    this->nodes[i-1]->op->generate_gradients(i-1, grad_messages);
                }
            }
            std::vector<NodeId> grads;
            for(int i=0;i<params.size();i++){
                grads.push_back(grad_messages[params[i]]);
            }
            return grads;
        }
        std::array<SymInt,4> shape(NodeId node){
            return this->nodes[node]->shape;
        };

        NodeId add(NodeId arg1, NodeId arg2);
        NodeId neg(NodeId arg1);
        NodeId mul(NodeId arg1_id, NodeId arg2_id);

        NodeId scalar(std::string name, ad_value_type v_type){
            std::array<SymInt,4> shape {SymInt::one(), SymInt::one(), SymInt::one(), SymInt::one()};
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    name,
                    ad_node_type::INPUT ,
                    v_type ,
                    shape,
                    std::make_shared<InputOperator>(this),
                    0
            );
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId scalar(ad_value_type v_type){
            return scalar("InputNode", v_type);
        }

        NodeId scalar_as(std::string name, NodeId node){
            return scalar(name, this->nodes[node]->v_type);
        }

        NodeId scalar_as(NodeId node){
            return scalar("InputNode", this->nodes[node]->v_type);
        }

        NodeId vector(std::string name, ad_value_type v_type, SymInt shape0){
            std::array<SymInt,4> shape {shape0, SymInt::one(), SymInt::one(), SymInt::one()};
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    name,
                    ad_node_type::INPUT ,
                    v_type ,
                    shape,
                    std::make_shared<InputOperator>(this),
                    0
            );
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId vector(std::string name, ad_value_type v_type, size_t shape0) {
            return vector(name, v_type, SymInt::as_polynomial(shape0));
        }

        NodeId vector(ad_value_type v_type, SymInt shape0){
            return vector("InputNode", v_type, shape0);
        }

        NodeId vector(ad_value_type v_type, size_t shape0){
            return vector("InputNode", v_type, shape0);
        }

        NodeId vector(std::string name, ad_value_type v_type){
            auto shape0 = this->get_new_symbolic_integer();
            return vector(name, v_type, shape0);
        }

        NodeId vector(ad_value_type v_type){
            auto shape0 = this->get_new_symbolic_integer();
            return vector("InputNode", v_type, shape0);
        }

        NodeId vector_as(std::string name, NodeId node){
            return vector(name, this->nodes[node]->v_type, this->nodes[node]->shape[0]);
        }

        NodeId vector_as(NodeId node){
            return vector("InputNode", this->nodes[node]->v_type, this->nodes[node]->shape[0]);
        }

        NodeId matrix(std::string name, ad_value_type v_type, SymInt shape0, SymInt shape1){
            std::array<SymInt,4> shape {shape0, shape1, SymInt::one(), SymInt::one()};
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    name,
                    ad_node_type::INPUT ,
                    v_type ,
                    shape,
                    std::make_shared<InputOperator>(this),
                    0
            );
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId matrix(std::string name, ad_value_type v_type, SymInt shape0, size_t shape1) {
            return matrix(name, v_type, shape0, SymInt::as_polynomial(shape1));
        }

        NodeId matrix(std::string name, ad_value_type v_type, size_t shape0, SymInt shape1) {
            return matrix(name, v_type, SymInt::as_polynomial(shape0), shape1);
        }

        NodeId matrix(std::string name, ad_value_type v_type, size_t shape0, size_t shape1) {
            return matrix(name, v_type, SymInt::as_polynomial(shape0), SymInt::as_polynomial(shape1));
        }

        NodeId matrix(ad_value_type v_type, SymInt shape0, SymInt shape1){
            return matrix("InputNode", v_type, shape0, shape1);
        }

        NodeId matrix(ad_value_type v_type, SymInt shape0, size_t shape1){
            return matrix("InputNode", v_type, shape0, shape1);
        }

        NodeId matrix(ad_value_type v_type, size_t shape0, SymInt shape1){
            return matrix("InputNode", v_type, shape0, shape1);
        }

        NodeId matrix(ad_value_type v_type, size_t shape0, size_t shape1){
            return matrix("InputNode", v_type, shape0, shape1);
        }

        NodeId matrix(std::string name, ad_value_type v_type){
            auto shape0 = this->get_new_symbolic_integer();
            auto shape1 = this->get_new_symbolic_integer();
            return matrix(name, v_type, shape0, shape1);
        }

        NodeId matrix(ad_value_type v_type){
            auto shape0 = this->get_new_symbolic_integer();
            auto shape1 = this->get_new_symbolic_integer();
            return matrix("InputNode", v_type, shape0, shape1);
        }

        NodeId square_matrix(std::string name, ad_value_type v_type, SymInt shape) {
            return matrix(name, v_type, shape, shape);
        }

        NodeId square_matrix(std::string name, ad_value_type v_type, size_t shape) {
            return matrix(name, v_type, shape, shape);
        }

        NodeId square_matrix(ad_value_type v_type, SymInt shape){
            return matrix("InputNode", v_type, shape, shape);
        }

        NodeId square_matrix(ad_value_type v_type, size_t shape){
            return matrix("InputNode", v_type, shape, shape);
        }

        NodeId square_matrix(std::string name, ad_value_type v_type){
            auto shape = this->get_new_symbolic_integer();
            return matrix(name, v_type, shape, shape);
        }

        NodeId square_matrix(ad_value_type v_type){
            auto shape = this->get_new_symbolic_integer();
            return matrix("InputNode", v_type, shape, shape);
        }

        NodeId matrix_as(std::string name, NodeId node){
            return matrix(name, this->nodes[node]->v_type,
                          this->nodes[node]->shape[0], this->nodes[node]->shape[1]);
        }

        NodeId matrix_as(NodeId node){
            return matrix("InputNode", this->nodes[node]->v_type,
                          this->nodes[node]->shape[0], this->nodes[node]->shape[1]);
        }

        NodeId tensor3(std::string name, ad_value_type v_type, SymInt shape0, SymInt shape1, SymInt shape2){
            std::array<SymInt,4> shape {shape0, shape1, shape2, SymInt::one()};
            auto result = std::make_shared<Node>(
                    this,
                    default_device,
                    nodes.size(),
                    name,
                    ad_node_type::INPUT ,
                    v_type ,
                    shape,
                    std::make_shared<InputOperator>(this),
                    0
            );
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId tensor3(std::string name, ad_value_type v_type, SymInt shape0, SymInt shape1, size_t shape2) {
            return tensor3(name, v_type, shape0, shape1, SymInt::as_polynomial(shape2));
        }

        NodeId tensor3(std::string name, ad_value_type v_type, SymInt shape0, size_t shape1, SymInt shape2) {
            return tensor3(name, v_type, shape0, SymInt::as_polynomial(shape1), shape2);
        }

        NodeId tensor3(std::string name, ad_value_type v_type, SymInt shape0, size_t shape1, size_t shape2) {
            return tensor3(name, v_type, shape0, SymInt::as_polynomial(shape1), SymInt::as_polynomial(shape2));
        }

        NodeId tensor3(std::string name, ad_value_type v_type, size_t shape0, SymInt shape1, SymInt shape2) {
            return tensor3(name, v_type, SymInt::as_polynomial(shape0), shape1, shape2);
        }

        NodeId tensor3(std::string name, ad_value_type v_type, size_t shape0, SymInt shape1, size_t shape2) {
            return tensor3(name, v_type, SymInt::as_polynomial(shape0), shape1, SymInt::as_polynomial(shape2));
        }

        NodeId tensor3(std::string name, ad_value_type v_type, size_t shape0, size_t shape1, SymInt shape2) {
            return tensor3(name, v_type, SymInt::as_polynomial(shape0), SymInt::as_polynomial(shape1), shape2);
        }

        NodeId tensor3(std::string name, ad_value_type v_type, size_t shape0, size_t shape1, size_t shape2) {
            return tensor3(name, v_type, SymInt::as_polynomial(shape0), SymInt::as_polynomial(shape1), SymInt::as_polynomial(shape2));
        }

        NodeId tensor3(ad_value_type v_type, SymInt shape0, SymInt shape1, SymInt shape2){
            return tensor3("InputNode", v_type, shape0, shape1, shape2);
        }

        NodeId tensor3(ad_value_type v_type, SymInt shape0, SymInt shape1, size_t shape2) {
            return tensor3("InputNode", v_type, shape0, shape1, SymInt::as_polynomial(shape2));
        }

        NodeId tensor3(ad_value_type v_type, SymInt shape0, size_t shape1, SymInt shape2) {
            return tensor3("InputNode", v_type, shape0, SymInt::as_polynomial(shape1), shape2);
        }

        NodeId tensor3(ad_value_type v_type, SymInt shape0, size_t shape1, size_t shape2) {
            return tensor3("InputNode", v_type, shape0, SymInt::as_polynomial(shape1), SymInt::as_polynomial(shape2));
        }

        NodeId tensor3(ad_value_type v_type, size_t shape0, SymInt shape1, SymInt shape2) {
            return tensor3("InputNode", v_type, SymInt::as_polynomial(shape0), shape1, shape2);
        }

        NodeId tensor3(ad_value_type v_type, size_t shape0, SymInt shape1, size_t shape2) {
            return tensor3("InputNode", v_type, SymInt::as_polynomial(shape0), shape1, SymInt::as_polynomial(shape2));
        }

        NodeId tensor3(ad_value_type v_type, size_t shape0, size_t shape1, SymInt shape2) {
            return tensor3("InputNode", v_type, SymInt::as_polynomial(shape0), SymInt::as_polynomial(shape1), shape2);
        }

        NodeId tensor3(ad_value_type v_type, size_t shape0, size_t shape1, size_t shape2) {
            return tensor3("InputNode", v_type, SymInt::as_polynomial(shape0), SymInt::as_polynomial(shape1), SymInt::as_polynomial(shape2));
        }

        NodeId tensor3(std::string name, ad_value_type v_type) {
            auto shape0 = this->get_new_symbolic_integer();
            auto shape1 = this->get_new_symbolic_integer();
            auto shape2 = this->get_new_symbolic_integer();
            return tensor3(name, v_type, shape0, shape1, shape2);
        }

        NodeId tensor3(ad_value_type v_type) {
            auto shape0 = this->get_new_symbolic_integer();
            auto shape1 = this->get_new_symbolic_integer();
            auto shape2 = this->get_new_symbolic_integer();
            return tensor3("InputNode", v_type, shape0, shape1, shape2);
        }

        NodeId tensor3_as(std::string name, NodeId node){
            return tensor3(name, this->nodes[node]->v_type,
                           this->nodes[node]->shape[0],
                           this->nodes[node]->shape[1],
                           this->nodes[node]->shape[2]);
        }

        NodeId tensor3_as(NodeId node){
            return tensor3("InputNode", this->nodes[node]->v_type,
                           this->nodes[node]->shape[0],
                           this->nodes[node]->shape[1],
                           this->nodes[node]->shape[2]);
        }
    };

    class OperatorError : public std::exception{
    public:
        std::string name;
        std::vector<size_t> input_ids;
        std::vector<std::array<SymInt,4>> input_shapes;
        OperatorError(std::string name,
                      std::vector<std::weak_ptr<Node>> inputs):
                name(name)
        {
            for(int i=0;i < inputs.size(); i++){
                input_ids.push_back(inputs[i].lock()->id);
                input_shapes.push_back(inputs[i].lock()->shape);
            }
        };
    };

    class IncompatibleShapes: public OperatorError
    {
    public:
        IncompatibleShapes(std::string name,
                           std::vector<std::weak_ptr<Node>> inputs) :
                OperatorError(name, inputs)
        {}
        virtual const char* what() const throw()
        {
            std::string id_msg;
            std::string shape_msg;
            for(auto i=0;i<input_ids.size();i++){
                id_msg += std::to_string(input_ids[i]);
                shape_msg += "[";
                for(auto j=0; j<4; j++){
                    shape_msg += input_shapes[i][j].to_string();
                    if(j < 3){
                        shape_msg += ", ";
                    }
                }
                shape_msg += "]";
                if(i < input_ids.size() - 1){
                    id_msg += ", ";
                    shape_msg += ", ";
                }
            }
            std::string msg = "\nIncomaptible dimensions in operator '" + name + "'\n" +
                              "Input ids: " + id_msg + "\n" +
                              "Input shapes: " + shape_msg;
            return msg.c_str();
        }
    };

    class ImplicitBroadcast: public OperatorError
    {
    public:
        std::vector<size_t> dims;
        ImplicitBroadcast(std::string name,
                           std::vector<std::weak_ptr<Node>> inputs,
        std::vector<size_t> dims) :
                OperatorError(name, inputs),
                dims(dims)
        {}
        virtual const char* what() const throw()
        {
            std::string id_msg;
            std::string shape_msg;
            for(auto i=0;i<input_ids.size();i++){
                id_msg += std::to_string(input_ids[i]);
                shape_msg += "[";
                for(auto j=0; j<4; j++){
                    shape_msg += input_shapes[i][j].to_string();
                    if(j < 3){
                        shape_msg += ", ";
                    }
                }
                shape_msg += "]";
                if(i < input_ids.size() - 1){
                    id_msg += ", ";
                    shape_msg += ", ";
                }
            }
            std::string dim_msg;
            for(int i=0;i<dims.size();i++){
                dim_msg += std::to_string(dims[i]);
                if(i < dims.size() - 1){
                    dim_msg += ", ";
                }
            }
            std::string msg = "\nImplicit broadvast in operator '" + name + "' " +
                              "along dimensions " + dim_msg + "\n" +
                              "Input ids: " + id_msg + "\n" +
                              "Input shapes: " + shape_msg;
            return msg.c_str();
        }
    };
}

#endif //AUTODIFF_CORE_H