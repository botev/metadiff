//
// Created by alex on 10/12/15.
//

#ifndef METADIFF_CORE_H
#define METADIFF_CORE_H

namespace metadiff {
    const size_t N = 100;

    typedef symbolic::SymbolicPolynomial<N, unsigned short> SymInt;
    typedef std::array<SymInt,4> Shape;

    enum ad_node_type{SYMBOLIC_INTEGER, CONSTANT, INPUT, SHARED_INPUT, INPUT_DERIVED, CONSTANT_DERIVED};
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


//    class ConstValue{
//    public:
//        af::array value;
//        ConstValue(){};
//
//        ConstValue(void * pointer):
//                pointer(pointer),
//                array(true)
//        {};
//        ConstValue(double num_value):
//                num_value(num_value),
//                array(false)
//        {};
//    };

    class SharedVariable{
        // TODO properly
    public:
        Shape shape;
        SharedVariable(Shape shape):
                shape(shape)
        {};
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
        Operator(std::string name, GraphInPtr graph):
                name(name),
                graph(graph)
        {};

        virtual void generate_gradients(size_t current, std::unordered_map<size_t , size_t>& messages) = 0;
        virtual ad_value_type get_value_type() = 0;
        virtual Shape get_shape() = 0;
        virtual ad_node_type get_node_type() = 0;
        virtual unsigned short get_gradient_level() = 0;
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
        af::array value;

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

        void generate_gradients(std::unordered_map<size_t , size_t>& messages){
            this->op->generate_gradients(id, messages);
        }

        bool is_constant() const;

        bool is_scalar() const{
            for(int i=0; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_vector() const{
            for(int i=1; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_vector_strict() const{
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

        bool is_matrix() const{
            for(int i=2; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_matrix_strict() const{
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

        bool is_tensor3() const{
            for(int i=3; i < 4; i++){
                if(shape[i] != 1){
                    return false;
                }
            }
            return true;
        }

        bool is_tensor3_strict() const{
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

        bool is_tensor4_strict() const{
            for(int i=0; i < 4; i++){
                if(shape[i] == 1){
                    return false;
                }
            }
            return true;
        }
    };

    void setDevice(GraphInPtr graph, size_t id, Device device);

    class Node{
    public:
        GraphInPtr graph;
        size_t id;
        Node(GraphInPtr graph, size_t id):
                graph(graph),
                id(id)
        {};

        void setDevice(Device device){
            metadiff::setDevice(graph, id, device);
        }

        Shape shape();
        bool is_scalar();
        bool is_vector();
        bool is_vector_strict();
        bool is_matrix();
        bool is_matrix_strict();
        bool is_tensor3();
        bool is_tensor3_strict();
        bool is_tensor4_strict();

        Node sum(std::vector<size_t> axes={0,1,2,3});

        Node broadcast(Shape shape);
        Node broadcast_to(Node other);

        Node zeros();
        Node non_zeros();
        Node is_nan();
        Node is_inf();

        Node exp();
        Node log();
        Node pow();
        Node abs();
        Node sin();
        Node cos();
        Node tan();
        Node cot();
        Node sinh();
        Node cosh();
        Node tanh();
        Node coth();
        Node sigmoid();
        Node constant();

        Node transpose();
        Node diag();
        Node minv();
        Node det();
        Node logdet();
        Node trace();

        Node reshape(Shape shape);
        Node flatten(size_t ndim=1);
        Node reorder(std::array<size_t, 4> order);
        Node reorder(size_t dim1, size_t dim2, size_t dim3=2, size_t dim4=3);

        Node softplus(double threshold = 50);
    };

    class Input : public Operator {
    public:
        Input(GraphInPtr graph): Operator("Input", graph){}

        ad_value_type get_value_type(){
            return ad_value_type::FLOAT;
        }

        Shape get_shape(){
            return Shape{0, 0, 0, 0};
        }

        ad_node_type get_node_type(){
            return INPUT;
        };

        unsigned short get_gradient_level(){
            return 0;
        }

        NodeInVec get_parents(){
            return NodeInVec {};
        }

        NodeInVec get_arguments(){
            return NodeInVec {};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t>& messages);
    };

    class UnsupportedGradient : public std::exception {
        const char* what() const throw(){
            return "\nThe gradient operation supports only scalar values";
        }
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
        std::vector<size_t> temporary_constants;

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

        std::vector<bool> get_descendands_mask(std::vector<Node> marked){
            auto n = this->nodes.size();
            std::vector<bool> descendands_mask(n, false);
            for(int i=0;i<marked.size();i++){
                descendands_mask[marked[i].id] = true;
            }

            // Mark all direct children
            for(int i=0;i<n; i++){
                if(descendands_mask[i]){
                    auto children = nodes[i]->children;
                    for(int j=0;j<children.size();j++){
                        descendands_mask[children[j].lock()->id] = true;
                    }
                }
            }
            return descendands_mask;
        }

        std::vector<bool> get_ancestors_mask(std::vector<Node> marked){
            // Assumes that computations are ordered
            auto n = this->nodes.size();
            std::vector<bool> ancestors_mask(n, false);
            for(int i=0;i<marked.size();i++){
                ancestors_mask[marked[i].id] = true;
            }

            // Mark all direct ancesotrs
            for(int i=n-1;i >= 0; i--){
                if(ancestors_mask[i]){
                    auto ancestors = nodes[i]->op->get_ancestors();
                    for(int j=0;j<ancestors.size();j++){
                        ancestors_mask[ancestors[j].lock()->id] = true;
                    }
                }
            }

            return ancestors_mask;
        }

        std::vector<Node> gradient(Node objective, std::vector<Node> params) {
            if(not nodes[objective.id]->is_scalar()){
                throw UnsupportedGradient();
            }
            std::unordered_map<size_t, size_t> grad_messages;

            // Extract the flow tree between params and objective
            auto descendence_mask = get_descendands_mask(params);
            auto ancestors_mask = get_ancestors_mask({objective});
            std::vector<size_t> flow_tree;
            temporary_constants.clear();
            for(size_t i=0;i<nodes.size(); i++) {
                if(ancestors_mask[i] and descendence_mask[i]){
                    flow_tree.push_back(i);
                } else {
                    temporary_constants.push_back(i);
                }
            }
//            std::cout << "Temp consts: ";
//            for(int i=0;i<temporary_constants.size();i++){
//                std::cout << temporary_constants[i] << ", ";
//            }
//            std::cout << std::endl;

            // Send the first message as 1 to the objective
            auto target = this->nodes[objective.id];
            auto unity_grad = this->constant_node(af::constant(1.0, 1)).id;
            this->nodes[unity_grad]->grad_level = target->grad_level + ((unsigned short) 1);
            this->nodes[unity_grad]->name = "";
            grad_messages[target->id] = unity_grad;

            // Send all gradient messages
            for (auto i = flow_tree.size(); i > 0; i--) {
                auto node_id = flow_tree[i-1];
                if (grad_messages.find(node_id) != grad_messages.end()) {
                    this->nodes[node_id]->generate_gradients(grad_messages);
                }
            }

            // Extract the gradients for each parameter
            std::vector<Node> grads;
            for (int i = 0; i < params.size(); i++) {
                grads.push_back(Node(shared_from_this(), grad_messages[params[i].id]));
            }

            // Restore types of other inputs
            temporary_constants.clear();
            return grads;
        }

        int find_same_node(std::shared_ptr<Operator> op){
            return -1;
        }

        NodeInPtr derived_node(std::shared_ptr<Operator> op,
                               int grad_level = -1) {
            int same_node = find_same_node(op);
            if(same_node == -1) {
                auto parents = op->get_parents();
                grad_level = grad_level == -1 ? op->get_gradient_level() : grad_level;
                auto result = std::make_shared<NodeInternal>(
                        shared_from_this(),
                        default_device,
                        nodes.size(),
                        "Derived Node",
                        op->get_node_type(),
                        op->get_value_type(),
                        op->get_shape(),
                        op,
                        grad_level
                );
                this->nodes.push_back(result);
                NodeInVec ancestors = op->get_ancestors();
                for (int i = 0; i < ancestors.size(); i++) {
                    ancestors[i].lock()->children.push_back(result);
                }
                return result;
            } else {
                return nodes[same_node];
            }
        }

//        Node constant_node(double *value, std::array<size_t, 4> dims) {
//            auto result = std::make_shared<NodeInternal>(
//                    shared_from_this(),
//                    default_device,
//                    nodes.size(),
//                    "Constant Node",
//                    ad_node_type::CONSTANT,
//                    ad_value_type::FLOAT,
//                    Shape {dims[0], dims[1], dims[2], dims[3]},
//                    std::make_shared<Input>(shared_from_this()),
//                    0
//            );
//            result->value.pointer = value;
//            this->nodes.push_back(result);
//            return Node(shared_from_this(), result->id);
//        }
//
//        Node constant_node(double value) {
//            auto result = std::make_shared<NodeInternal>(
//                    shared_from_this(),
//                    default_device,
//                    nodes.size(),
//                    "Constant Node",
//                    ad_node_type::CONSTANT,
//                    ad_value_type::FLOAT,
//                    Shape {1, 1, 1, 1},
//                    std::make_shared<Input>(shared_from_this()),
//                    0
//            );
//            result->value.num_value = value;
//            this->nodes.push_back(result);
//            return Node(shared_from_this(), result->id);
//        }
//
//        Node constant_node(int *value, std::array<size_t, 4> dims) {
//            auto result = std::make_shared<NodeInternal>(
//                    shared_from_this(),
//                    default_device,
//                    nodes.size(),
//                    "Constant Node",
//                    ad_node_type::CONSTANT,
//                    ad_value_type::INTEGER,
//                    Shape {dims[0], dims[1], dims[2], dims[3]},
//                    std::make_shared<Input>(shared_from_this()),
//                    0
//            );
//            result->value.pointer = value;
//            this->nodes.push_back(result);
//            return Node(shared_from_this(), result->id);
//        }
//
//        Node constant_node(int value) {
//            auto result = std::make_shared<NodeInternal>(
//                    shared_from_this(),
//                    default_device,
//                    nodes.size(),
//                    "Constant Node",
//                    ad_node_type::CONSTANT,
//                    ad_value_type::INTEGER,
//                    Shape {1, 1, 1, 1},
//                    std::make_shared<Input>(shared_from_this()),
//                    0
//            );
//            result->value.num_value = value;
//            this->nodes.push_back(result);
//            return Node(shared_from_this(), result->id);
//        }
//
//        Node constant_node(bool *value, std::array<size_t, 4> dims) {
//            auto result = std::make_shared<NodeInternal>(
//                    shared_from_this(),
//                    default_device,
//                    nodes.size(),
//                    "Constant Node",
//                    ad_node_type::CONSTANT,
//                    ad_value_type::INTEGER,
//                    Shape {dims[0], dims[1], dims[2], dims[3]},
//                    std::make_shared<Input>(shared_from_this()),
//                    0
//            );
//            result->value.pointer = value;
//            this->nodes.push_back(result);
//            return Node(shared_from_this(), result->id);
//        }
//
//        Node constant_node(bool value) {
//            auto result = std::make_shared<NodeInternal>(
//                    shared_from_this(),
//                    default_device,
//                    nodes.size(),
//                    "Constant Node",
//                    ad_node_type::CONSTANT,
//                    ad_value_type::INTEGER,
//                    Shape {1, 1, 1, 1},
//                    std::make_shared<Input>(shared_from_this()),
//                    0
//            );
//            result->value.num_value = value;
//            this->nodes.push_back(result);
//            return Node(shared_from_this(), result->id);
//        }
        Node constant_node(af::array value){
            ad_value_type dtype;
            if(value.type() == af::dtype::b8){
                dtype = BOOLEAN;
            }
            if(value.type() == af::dtype::f32
                    or value.type() == af::dtype::f64){
                dtype = FLOAT;
            } else {
                dtype = INTEGER;
            }
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
                    default_device,
                    nodes.size(),
                    "Constant Node",
                    ad_node_type::CONSTANT,
                    dtype,
                    Shape {1, 1, 1, 1},
                    std::make_shared<Input>(shared_from_this()),
                    0
            );
            result->value = value;
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
                    std::make_shared<Input>(shared_from_this()),
                    0
            );
            this->nodes.push_back(result);
            return Node(shared_from_this(), result->id);
        }

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

        Node vector(ad_value_type v_type,
                    SymInt shape,
                    std::string name = "InputVector") {
            std::array<SymInt, 4> shape_t{shape, SymInt::one(), SymInt::one(), SymInt::one()};
            return tensor(v_type, shape_t, name);
        }

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

    void Input::generate_gradients(size_t current, std::unordered_map<size_t, size_t>& messages){
        auto graph = this->graph.lock();
        if(messages.find(current) == messages.end()){
            return;
        }
        auto my_grad = graph->nodes[messages[current]];
        if(my_grad->name == "Derived Node"){
            my_grad->name = "Grad of " + std::to_string(current);
        } else {
            my_grad->name += "|Grad of " + std::to_string(current);
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

    void setDevice(GraphInPtr graph, size_t id, Device device){
        graph.lock()->nodes[id]->device = device;
    }

    bool NodeInternal::is_constant() const{
        auto consts = graph.lock()->temporary_constants;
        return type == ad_node_type::CONSTANT
               or type == ad_node_type::SYMBOLIC_INTEGER
               or type == ad_node_type::CONSTANT_DERIVED
               or std::find(consts.begin(), consts.end(),  id) != consts.end();
    }

// <broadcast 1, broadcast 2>


//    std::array<bool,2> verify_shapes(NodeInPtr node0_ptr, NodeInPtr node1_ptr){
//        auto node0 = node0_ptr.lock();
//        auto node1 = node1_ptr.lock();
//        if(node0->shape == node1->shape or node0->is_scalar() or node1->is_scalar()){
//            return {false, false};
//        }
//        bool broadcast0 = false;
//        bool broadcast1 = false;
//        for(int i=0; i<4; i++){
//            if(node0->shape[i] == node1->shape[i]){
//                continue;
//            } else if(node0->shape[i] == 1 and node1->shape[i] != 1){
//                broadcast0 = true;
//            } else if(node0->shape[i] != 1 and node1->shape[i] == 1){
//                broadcast1 = true;
//            } else {
//                return {true, true};
//            }
//        }
//        return {broadcast0, broadcast1};
//    }

    class OperatorError : public std::exception{
    public:
        std::string name;
        std::vector<size_t> input_ids;
        std::vector<Shape> input_shapes;
        OperatorError(std::string name,
                      std::vector<size_t> input_ids,
                      std::vector<Shape> input_shapes):
                name(name),
                input_ids(input_ids),
                input_shapes(input_shapes)
        {};
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
                           std::vector<size_t> input_ids,
                           std::vector<Shape> input_shapes):
                OperatorError(name, input_ids, input_shapes)
        {}

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
                if(i < input_ids.size() - 1){
                    id_msg += ", ";
                }
            }
            for(auto i=0;i<input_shapes.size();i++){
                shape_msg += "[";
                for(auto j=0; j<4; j++){
                    shape_msg += input_shapes[i][j].to_string();
                    if(j < 3){
                        shape_msg += ", ";
                    }
                }
                shape_msg += "]";
                if(i < input_ids.size() - 1){
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
        ImplicitBroadcast(std::string name,
                          std::vector<size_t> input_ids,
                          std::vector<Shape> input_shapes):
                OperatorError(name, input_ids, input_shapes)
        {}

        ImplicitBroadcast(std::string name,
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
//            std::string dim_msg;
//            for(int i=0;i<dims.size();i++){
//                dim_msg += std::to_string(dims[i]);
//                if(i < dims.size() - 1){
//                    dim_msg += ", ";
//                }
//            }
            std::string msg = "\nImplicit broadcast in operator '" + name + "'\n" +
                              //                              "along dimensions " + dim_msg + "\n" +
                              "Input ids: " + id_msg + "\n" +
                              "Input shapes: " + shape_msg;
            return msg.c_str();
        }
    };

    class InvalidArguments: public OperatorError
    {
    public:
        std::string argument_string;
        InvalidArguments(std::string name,
                         std::vector<size_t> input_ids,
                         std::vector<Shape> input_shapes,
                         std::string argument_string):
                OperatorError(name, input_ids, input_shapes),
                argument_string(argument_string)
        {};

        InvalidArguments(std::string name,
                         NodeInVec inputs,
                         std::string argument_string):
                OperatorError(name, inputs),
                argument_string(argument_string)
        {};

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
            std::string msg = "\nInvalid arguments in operator '" + name + "'\n" +
                              "Input ids: " + id_msg + "\n" +
                              "Input shapes: " + shape_msg + "\n" +
                              "Arguments: " + argument_string + "\n";
            return msg.c_str();
        }
    };

    class UnknownError: public OperatorError
    {
    public:
        std::string err_string;
        UnknownError(std::string name,
                     std::vector<size_t> input_ids,
                     std::vector<Shape> input_shapes,
                     std::string err_string):
                OperatorError(name, input_ids, input_shapes),
                err_string(err_string)
        {};

        UnknownError(NodeInVec inputs,
                     std::string err_string):
                OperatorError(name, inputs),
                err_string(err_string)
        {};

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
            std::string msg = "\nUnkown error in operator '" + name + "'\n" +
                              "Input ids: " + id_msg + "\n" +
                              "Input shapes: " + shape_msg + "\n" +
                              "Error message: " + err_string + "\n";
            return msg.c_str();
        }
    };

    std::string to_string(ad_node_type const & type){
        switch(type){
            case ad_node_type::SYMBOLIC_INTEGER: return "SYMBOLIC_INTEGER";
            case ad_node_type::CONSTANT: return "CONSTANT";
            case ad_node_type::INPUT : return "INPUT";
            case ad_node_type::SHARED_INPUT : return "SHARED";
            case ad_node_type::INPUT_DERIVED: return "DERIVED";
            case ad_node_type::CONSTANT_DERIVED: return "CONSTANT_DERIVED";

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
        return to_string(device.type) + "[" + std::to_string(device.id) + "]";
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
        f << to_string(device);
        return f;
    }
}

#endif //METADIFF_CORE_H