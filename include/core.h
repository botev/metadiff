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

namespace autodiff {
    const size_t N = 100;

    typedef symbolic::SymbolicPolynomial<N, unsigned short> SymInt;
    typedef std::array<SymInt,4> Shape;

    enum ad_node_type{CONSTANT, INPUT, SHARED_INPUT, INPUT_DERIVED, SYMBOLIC_INTEGER};
    enum ad_value_type{FLOAT, INTEGER, BOOLEAN};
    enum ad_device_type {CPU, GPU};
    enum ad_implicit_broadcast {RAISE, WARN, QUIET};
    enum ad_float_type {f16, c16, f32, c32, f64, c64};
    enum ad_integer_type {s8, u8, s16, u16, s32, u32, s64, u64};

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

    class ConstValue{
    public:
        // For different type of values
        void* pointer;
        double num_value;
        // Symbolic value
        SymInt value;
        ConstValue(){};
        ConstValue(void * pointer):
                pointer(pointer)
        {};
        ConstValue(double num_value):
                num_value(num_value)
        {};
        ConstValue(SymInt value):
                value(value)
        {};
    };

    class SharedVariable{
    public:
        SharedVariable(){};
    };

    class GraphInternal;
    class NodeInternal;
    typedef std::weak_ptr<NodeInternal> NodeInPtr;
    typedef std::vector<NodeInPtr> NodeInVec;
    typedef std::weak_ptr<GraphInternal> GraphInPtr;
    typedef std::shared_ptr<GraphInternal> Graph;
    Graph create_graph(){
        return std::make_shared<GraphInternal>();
    }

    class Operator{
    public:
        GraphInPtr graph;
        const std::string name;
        Operator(GraphInPtr graph, std::string name):
                graph(graph),
                name(name)
        {};

        virtual void generate_gradients(size_t current, std::unordered_map<size_t , size_t>& messages) = 0;
        virtual ad_value_type get_value_type() = 0;
        virtual unsigned short get_gradient_level() = 0;
        virtual Shape get_shape() = 0;
        virtual NodeInVec get_parents() = 0;
        virtual NodeInVec get_arguments() = 0;

        NodeInVec get_ancestors(){
            auto parents = this->get_parents();
            auto arguments = this->get_arguments();
            for(int i=0; i<arguments.size();i++){
                parents.push_back(arguments[i]);
            }
            return parents;
        }
    };

    class NodeInternal{
    public:
        GraphInPtr graph;
        Device device;
        size_t id;
        std::string name;
        ad_node_type type;
        ad_value_type v_type;
        Shape shape;
        std::shared_ptr<Operator> op;
        NodeInVec children;
        unsigned short grad_level;
        ConstValue value;

        NodeInternal(GraphInPtr graph, Device device):
                graph(graph),
                device(device)
        {}

        NodeInternal(GraphInPtr graph,
                     Device device,
                     size_t id,
                     std::string name,
                     ad_node_type type,
                     ad_value_type v_type,
                     Shape shape,
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
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_vector(){
            for(int i=1; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_vector_strict(){
            for(int i=0; i < 1; i++){
                if(shape[i] == 1){
                    return false;
                }
            }
            for(int i=1; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_matrix(){
            for(int i=2; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_matrix_strict(){
            for(int i=0; i < 2; i++){
                if(shape[i] == 1){
                    return false;
                }
            }
            for(int i=2; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_tensor3(){
            for(int i=3; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_tensor3_strict(){
            for(int i=0; i < 3; i++){
                if(shape[i] == 1){
                    return false;
                }
            }
            for(int i=3; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_tensor4_strict(){
            for(int i=0; i < 4; i++){
                if(shape[i] == 1){
                    return false;
                }
            }
            return true;
        }
    };

    class Node{
    public:
        GraphInPtr graph;
        size_t id;
        Node(GraphInPtr graph, size_t id):
                graph(graph),
                id(id)
        {};

        Shape shape();
        bool is_scalar();
        bool is_vector();
        bool is_vector_strict();
        bool is_matrix();
        bool is_matrix_strict();
        bool is_tensor3();
        bool is_tensor3_strict();
        bool is_tensor4_strict();
    };

    class InputOperator : public Operator {
    public:
        InputOperator(GraphInPtr graph): Operator(graph, "Input"){}

        ad_value_type get_value_type(){
            return ad_value_type::FLOAT;
        }

        unsigned short get_gradient_level(){
            return 0;
        }

        Shape get_shape(){
            return Shape{0, 0, 0, 0};
        }

        NodeInVec get_parents(){
            return NodeInVec {};
        }

        NodeInVec get_arguments(){
            return NodeInVec {};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t>& messages){};
    };

    class GraphInternal : public std::enable_shared_from_this<GraphInternal> {
    public:
        std::vector<std::shared_ptr<NodeInternal>> nodes;
        std::string name;
        Device default_device;
        ad_float_type f_type;
        ad_integer_type i_type;
        ad_implicit_broadcast broadcast;
        size_t sym_integer_count;

        GraphInternal() {
            // TODO Have a better preference of devices available in order
            name = "Function";
            sym_integer_count = 0;
            default_device = Device(CPU, 0);
            f_type = ad_float_type::f32;
            i_type = ad_integer_type::s32;
            broadcast = ad_implicit_broadcast::RAISE;
        }

        SymInt get_new_symbolic_integer() {
            this->sym_integer_count++;
            return SymInt::variable(this->sym_integer_count - 1);
        }

        std::vector<Node> gradient(Node objective, std::vector<Node> params) {
            std::unordered_map<size_t, size_t> grad_messages;
            auto target = this->nodes[objective.id];
            long n = this->nodes.size();
            auto unity_grad = this->constant_node(1).id;
            this->nodes[unity_grad]->grad_level = target->grad_level + ((unsigned short) 1);
            grad_messages[target->id] = unity_grad;
            this->nodes[grad_messages[target->id]]->name = "Grad of " + std::to_string(objective.id);
            for (auto i = n; i > 0; i--) {
                if (grad_messages.find(i - 1) != grad_messages.end()) {
                    this->nodes[i - 1]->op->generate_gradients(i - 1, grad_messages);
                }
            }
            std::vector<Node> grads;
            for (int i = 0; i < params.size(); i++) {
                grads.push_back(Node(shared_from_this(), grad_messages[params[i].id]));
            }
            return grads;
        }

        size_t derived_node(std::shared_ptr<Operator> op) {
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
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
            NodeInVec ancestors = op->get_ancestors();
            for (int i = 0; i < ancestors.size(); i++) {
                ancestors[i].lock()->children.push_back(result);
            }
            return result->id;
        }

        Node constant_node(double *value, std::array<size_t, 4> dims) {
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT,
                    ad_value_type::FLOAT,
                    Shape {dims[0], dims[1], dims[2], dims[3]},
                    std::make_shared<InputOperator>(shared_from_this()),
                    0
            );
            result->value.pointer = value;
            this->nodes.push_back(result);
            return Node(shared_from_this(), result->id);
        }

        Node constant_node(double value) {
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT,
                    ad_value_type::FLOAT,
                    Shape {1, 1, 1, 1},
                    std::make_shared<InputOperator>(shared_from_this()),
                    0
            );
            result->value.num_value = value;
            this->nodes.push_back(result);
            return Node(shared_from_this(), result->id);
        }

        Node constant_node(int *value, std::array<size_t, 4> dims) {
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT,
                    ad_value_type::INTEGER,
                    Shape {dims[0], dims[1], dims[2], dims[3]},
                    std::make_shared<InputOperator>(shared_from_this()),
                    0
            );
            result->value.pointer = value;
            this->nodes.push_back(result);
            return Node(shared_from_this(), result->id);
        }

        Node constant_node(int value) {
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT,
                    ad_value_type::INTEGER,
                    Shape {1, 1, 1, 1},
                    std::make_shared<InputOperator>(shared_from_this()),
                    0
            );
            result->value.num_value = value;
            this->nodes.push_back(result);
            return Node(shared_from_this(), result->id);
        }

        Node constant_node(bool *value, std::array<size_t, 4> dims) {
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT,
                    ad_value_type::INTEGER,
                    Shape {dims[0], dims[1], dims[2], dims[3]},
                    std::make_shared<InputOperator>(shared_from_this()),
                    0
            );
            result->value.pointer = value;
            this->nodes.push_back(result);
            return Node(shared_from_this(), result->id);
        }

        Node constant_node(bool value) {
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT,
                    ad_value_type::INTEGER,
                    Shape {1, 1, 1, 1},
                    std::make_shared<InputOperator>(shared_from_this()),
                    0
            );
            result->value.num_value = value;
            this->nodes.push_back(result);
            return Node(shared_from_this(), result->id);
        }

        Node tensor(ad_value_type v_type,
                    std::array<SymInt, 4> shape,
                    std::string name = "InputTensor") {
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
                    default_device,
                    nodes.size(),
                    name,
                    ad_node_type::INPUT,
                    v_type,
                    shape,
                    std::make_shared<InputOperator>(shared_from_this()),
                    0
            );
            this->nodes.push_back(result);
            return Node(shared_from_this(), result->id);
        }

//        Node tensor(ad_value_type v_type,
//                    std::array<size_t, 4> shape,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape_t{shape[0], shape[1], shape[2], SymInt::one()};
//            return tensor(v_type, shape_t, name);
//        }

        Node tensor(ad_value_type v_type,
                    SymInt shape0,
                    SymInt shape1,
                    SymInt shape2,
                    SymInt shape3,
                    std::string name = "InputTensor") {
            std::array<SymInt, 4> shape{shape0,
                                        shape1,
                                        shape2,
                                        shape3};
            return tensor(v_type, shape, name);
        }

//        Node tensor(ad_value_type v_type,
//                    SymInt shape0,
//                    SymInt shape1,
//                    SymInt shape2,
//                    size_t shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{shape0,
//                                        shape1,
//                                        shape2,
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    SymInt shape0,
//                    SymInt shape1,
//                    size_t shape2,
//                    SymInt shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{shape0,
//                                        shape1,
//                                        shape3,
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    SymInt shape0,
//                    SymInt shape1,
//                    size_t shape2,
//                    size_t shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{shape0,
//                                        shape1,
//                                        shape2,
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    SymInt shape0,
//                    size_t shape1,
//                    SymInt shape2,
//                    SymInt shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{shape0,
//                                        shape1,
//                                        shape2,
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    SymInt shape0,
//                    size_t shape1,
//                    SymInt shape2,
//                    size_t shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{shape0,
//                                        shape1,
//                                        shape2,
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    SymInt shape0,
//                    size_t shape1,
//                    size_t shape2,
//                    SymInt shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{shape0,
//                                        shape1,
//                                        shape2,
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    SymInt shape0,
//                    size_t shape1,
//                    size_t shape2,
//                    size_t shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{shape0,
//                                        shape1,
//                                        SymInt::as_polynomial(shape2),
//                                        SymInt::as_polynomial(shape3)};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    size_t shape0,
//                    SymInt shape1,
//                    SymInt shape2,
//                    SymInt shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{SymInt::as_polynomial(shape0),
//                                        shape1,
//                                        shape2,
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    size_t shape0,
//                    SymInt shape1,
//                    SymInt shape2,
//                    size_t shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{SymInt::as_polynomial(shape0),
//                                        shape1,
//                                        shape2,
//                                        SymInt::as_polynomial(shape3)};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    size_t shape0,
//                    SymInt shape1,
//                    size_t shape2,
//                    SymInt shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{SymInt::as_polynomial(shape0),
//                                        shape1,
//                                        SymInt::as_polynomial(shape2),
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    size_t shape0,
//                    SymInt shape1,
//                    size_t shape2,
//                    size_t shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{SymInt::as_polynomial(shape0),
//                                        shape1,
//                                        SymInt::as_polynomial(shape2),
//                                        SymInt::as_polynomial(shape3)};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    size_t shape0,
//                    size_t shape1,
//                    SymInt shape2,
//                    SymInt shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{SymInt::as_polynomial(shape0),
//                                        SymInt::as_polynomial(shape1),
//                                        shape2,
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    size_t shape0,
//                    size_t shape1,
//                    SymInt shape2,
//                    size_t shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{SymInt::as_polynomial(shape0),
//                                        SymInt::as_polynomial(shape1),
//                                        shape2,
//                                        SymInt::as_polynomial(shape3)};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    size_t shape0,
//                    size_t shape1,
//                    size_t shape2,
//                    SymInt shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{SymInt::as_polynomial(shape0),
//                                        SymInt::as_polynomial(shape1),
//                                        SymInt::as_polynomial(shape2),
//                                        shape3};
//            return tensor(v_type, shape, name);
//        }
//
//        Node tensor(ad_value_type v_type,
//                    size_t shape0,
//                    size_t shape1,
//                    size_t shape2,
//                    size_t shape3,
//                    std::string name = "InputTensor") {
//            std::array<SymInt, 4> shape{SymInt::as_polynomial(shape0),
//                                        SymInt::as_polynomial(shape1),
//                                        SymInt::as_polynomial(shape2),
//                                        SymInt::as_polynomial(shape3)};
//            return tensor(v_type, shape, name);
//        }

        Node tensor(ad_value_type v_type,
                    std::string name = "InputTensor") {
            std::array<SymInt, 4> shape = {
                    get_new_symbolic_integer(),
                    get_new_symbolic_integer(),
                    get_new_symbolic_integer(),
                    get_new_symbolic_integer()
            };
            return tensor(v_type, shape, name);
        }

        Node tensor_as(Node node,
                       std::string name = "InputTensor") {
            return tensor(this->nodes[node.id]->v_type, node.shape(), name);
        }

        Node tensor3(ad_value_type v_type,
                     std::array<SymInt, 3> shape,
                     std::string name = "InputTensor3") {
            return tensor(v_type, {shape[0], shape[1], shape[2], 1}, name);
        }

//        Node tensor3(ad_value_type v_type,
//                     std::array<size_t, 3> shape,
//                     std::string name = "InputTensor3") {
//            std::array<SymInt, 4> shape_t{shape[0], shape[1], shape[2], SymInt::one()};
//            return tensor(v_type, shape_t, name);
//        }

        Node tensor3(ad_value_type v_type,
                     SymInt shape0,
                     SymInt shape1,
                     SymInt shape2,
                     std::string name = "InputTensor3") {
            return tensor(v_type, std::array<SymInt, 4>{
                                  shape0,
                                  shape1,
                                  shape2,
                                  1
                          },
                          name);
        }

//        Node tensor3(ad_value_type v_type,
//                     SymInt shape0,
//                     SymInt shape1,
//                     size_t shape2,
//                     std::string name = "InputTensor3") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  shape2,
//                                  SymInt::one()
//                          },
//                          name);
//        }
//
//        Node tensor3(ad_value_type v_type,
//                     SymInt shape0,
//                     size_t shape1,
//                     SymInt shape2,
//                     std::string name = "InputTensor3") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  shape2,
//                                  SymInt::one()
//                          },
//                          name);
//        }
//
//        Node tensor3(ad_value_type v_type,
//                     SymInt shape0,
//                     size_t shape1,
//                     size_t shape2,
//                     std::string name = "InputTensor3") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  shape2,
//                                  SymInt::one()
//                          },
//                          name);
//        }
//
//        Node tensor3(ad_value_type v_type,
//                     size_t shape0,
//                     SymInt shape1,
//                     SymInt shape2,
//                     std::string name = "InputTensor3") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  shape2,
//                                  SymInt::one()
//                          },
//                          name);
//        }
//
//        Node tensor3(ad_value_type v_type,
//                     size_t shape0,
//                     SymInt shape1,
//                     size_t shape2,
//                     std::string name = "InputTensor3") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  shape2,
//                                  SymInt::one()
//                          },
//                          name);
//        }
//
//        Node tensor3(ad_value_type v_type,
//                     size_t shape0,
//                     size_t shape1,
//                     SymInt shape2,
//                     std::string name = "InputTensor3") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  shape2,
//                                  SymInt::one()
//                          },
//                          name);
//        }
//
//        Node tensor3(ad_value_type v_type,
//                     size_t shape0,
//                     size_t shape1,
//                     size_t shape2,
//                     std::string name = "InputTensor3") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  shape2,
//                                  SymInt::one()
//                          },
//                          name);
//        }


        Node tensor3(ad_value_type v_type,
                     std::string name = "InputTensor3") {
            auto shape0 = this->get_new_symbolic_integer();
            auto shape1 = this->get_new_symbolic_integer();
            auto shape2 = this->get_new_symbolic_integer();
            return tensor3(v_type, shape0, shape1, shape2, name);
        }


        Node tensor3_as(Node node,
                        std::string name = "InputTensor3") {
            if(not node.is_tensor3()){
                throw "Node with id '" + std::to_string(node.id) + "' is not a tensor3.";
            }
            return tensor3(this->nodes[node.id]->v_type,
                           this->nodes[node.id]->shape[0],
                           this->nodes[node.id]->shape[1],
                           this->nodes[node.id]->shape[2],
                           name);
        }

        Node matrix(ad_value_type v_type,
                    std::array<SymInt, 2> shape,
                    std::string name = "InputMatrix") {
            std::array<SymInt, 4> shape_t{shape[0], shape[1], SymInt::one(), SymInt::one()};
            return tensor(v_type, shape_t, name);
        }

//        Node matrix(ad_value_type v_type,
//                    std::array<size_t, 2> shape,
//                    std::string name = "InputMatrix") {
//            std::array<SymInt, 4> shape_t{shape[0], shape[1], SymInt::one(), SymInt::one()};
//            return tensor(v_type, shape_t, name);
//        }

        Node matrix(ad_value_type v_type,
                    SymInt shape0,
                    SymInt shape1,
                    std::string name = "InputMatrix") {
            return tensor(v_type, std::array<SymInt, 4>{
                                  shape0,
                                  shape1,
                                  SymInt::one(),
                                  SymInt::one()
                          },
                          name);
        }

//        Node matrix(ad_value_type v_type,
//                    SymInt shape0,
//                    size_t shape1,
//                    std::string name = "InputMatrix") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  SymInt::one(),
//                                  SymInt::one()
//                          },
//                          name);
//        }
//
//        Node matrix(ad_value_type v_type,
//                    size_t shape0,
//                    SymInt shape1,
//                    std::string name = "InputMatrix") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  SymInt::one(),
//                                  SymInt::one()
//                          },
//                          name);
//        }
//
//        Node matrix(ad_value_type v_type,
//                    size_t shape0,
//                    size_t shape1,
//                    std::string name = "InputMatrix") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  shape1,
//                                  SymInt::one(),
//                                  SymInt::one()
//                          },
//                          name);
//        }

        Node matrix(ad_value_type v_type,
                    std::string name = "InputMatrix") {
            auto shape0 = this->get_new_symbolic_integer();
            auto shape1 = this->get_new_symbolic_integer();
            return matrix(v_type, shape0, shape1, name);
        }


        Node matrix_as(Node node,
                       std::string name = "InputMatrix") {
            if(not node.is_matrix()){
                throw "Node with id '" + std::to_string(node.id) + "' is not a matrix.";
            }
            return matrix(this->nodes[node.id]->v_type,
                          this->nodes[node.id]->shape[0],
                          this->nodes[node.id]->shape[1],
                          name);
        }

        Node square_matrix(ad_value_type v_type,
                           SymInt shape,
                           std::string name = "InputMatrix") {
            std::array<SymInt, 4> shape_t{shape, shape, SymInt::one(), SymInt::one()};
            return tensor(v_type, shape_t, name);
        }

//        Node square_matrix(ad_value_type v_type,
//                           size_t shape,
//                           std::string name = "InputMatrix") {
//            std::array<SymInt, 4> shape_t{shape, shape, SymInt::one(), SymInt::one()};
//            return tensor(v_type, shape_t, name);
//        }

        Node vector(ad_value_type v_type,
                    SymInt shape,
                    std::string name = "InputVector") {
            std::array<SymInt, 4> shape_t{shape, SymInt::one(), SymInt::one(), SymInt::one()};
            return tensor(v_type, shape_t, name);
        }

//        Node vector(ad_value_type v_type,
//                    size_t shape0,
//                    std::string name = "InputVector") {
//            return tensor(v_type, std::array<SymInt, 4>{
//                                  shape0,
//                                  SymInt::one(),
//                                  SymInt::one(),
//                                  SymInt::one()
//                          },
//                          name);
//        }

        Node vector(ad_value_type v_type,
                    std::string name = "InputVector") {
            auto shape0 = this->get_new_symbolic_integer();
            return vector(v_type, shape0, name);
        }


        Node vector_as(Node node,
                       std::string name = "InputVector") {
            if(not node.is_vector()){
                throw "Node with id '" + std::to_string(node.id) + "' is not a vector.";
            }
            return vector(this->nodes[node.id]->v_type,
                          this->nodes[node.id]->shape[0],
                          name);
        }

        Node scalar(ad_value_type v_type,
                    std::string name = "InputScalar") {
            std::array<SymInt, 4> shape_t{SymInt::one(), SymInt::one(), SymInt::one(), SymInt::one()};
            return tensor(v_type, shape_t, name);
        }
    };

    Shape Node::shape(){
        return this->graph.lock()->nodes[id]->shape;
    }

    bool Node::is_vector(){
        return this->graph.lock()->nodes[id]->is_vector();
    }

    bool Node::is_vector_strict(){
        return this->graph.lock()->nodes[id]->is_vector_strict();
    }

    bool Node::is_matrix(){
        return this->graph.lock()->nodes[id]->is_matrix();
    }


    bool Node::is_matrix_strict(){
        return this->graph.lock()->nodes[id]->is_matrix_strict();
    }

    bool Node::is_tensor3(){
        return this->graph.lock()->nodes[id]->is_tensor3();
    }

    bool Node::is_tensor3_strict(){
        return this->graph.lock()->nodes[id]->is_tensor3_strict();
    }

    bool Node::is_tensor4_strict(){
        return this->graph.lock()->nodes[id]->is_tensor4_strict();
    }

    class OperatorError : public std::exception{
    public:
        std::string name;
        std::vector<size_t> input_ids;
        std::vector<Shape> input_shapes;
        OperatorError(std::string name,
                      NodeInVec inputs):
                name(name)
        {
            for(int i=0;i < inputs.size(); i++){
                input_ids.push_back(inputs[i].lock()->id);
                input_shapes.push_back(inputs[i].lock()->shape);
            }
        };
        virtual std::string get_message() const = 0;
        const char* what() const throw(){
            return this->get_message().c_str();
        }
    };

    class IncompatibleShapes: public OperatorError
    {
    public:
        IncompatibleShapes(std::string name,
                           NodeInVec inputs) :
                OperatorError(name, inputs)
        {}
        std::string get_message() const
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
                          NodeInVec inputs,
                          std::vector<size_t> dims) :
                OperatorError(name, inputs),
                dims(dims)
        {}
        std::string get_message() const
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
            std::string msg = "\nImplicit broadcast in operator '" + name + "' " +
                              "along dimensions " + dim_msg + "\n" +
                              "Input ids: " + id_msg + "\n" +
                              "Input shapes: " + shape_msg;
            return msg.c_str();
        }
    };

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

    std::string to_string(Device const & device){
        return autodiff::to_string(device.type) + "[" + std::to_string(device.id) + "]";
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

    std::ostream & operator<<(std::ostream & f, Device const & device) {
        f << autodiff::to_string(device);
        return f;
    }
}

#endif //AUTODIFF_CORE_H