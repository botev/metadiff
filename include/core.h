//
// Created by alex on 10/12/15.
//

#ifndef METADIFF_CORE_H
#define METADIFF_CORE_H

//class SharedVariable{
//public:
//    size_t id;
//    af::array value;
//    SharedVariable():
//            id(0),
//            value(af::array())
//    {};
//    SharedVariable(size_t id, af::array value):
//            id(id),
//            value(value)
//    {};
//};
//
//typedef std::shared_ptr<SharedVariable> SharedPtr;

namespace metadiff {
    const size_t N = 100;

    typedef symbolic::SymbolicPolynomial<N, unsigned short> SymInt;
    typedef std::array<SymInt,4> Shape;

    enum ad_node_type{SYMBOLIC_INTEGER, CONSTANT, INPUT, SHARED_INPUT, INPUT_DERIVED, CONSTANT_DERIVED, UPDATE};
    enum ad_value_type{FLOAT, INTEGER, BOOLEAN};
    enum ad_device_type {CPU, GPU};
    enum ad_implicit_broadcast {RAISE, WARN, QUIET};
    enum ad_float_type {f16, c16, f32, c32, f64, c64};
    enum ad_integer_type {s8, u8, s16, u16, s32, u32, s64, u64};

    class GraphInternal;
    typedef GraphInternal* GraphInPtr;
    typedef std::shared_ptr<GraphInternal> Graph;

    Graph create_graph(){
        return std::make_shared<GraphInternal>();
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

    class ExecutionData{
    public:
        bool inlined;
        size_t register_id;
        size_t lifespan;
        ExecutionData():
                inlined(false),
                register_id(0),
                lifespan(0)
        {}
    };

    class Operator{
    public:
        GraphInPtr graph;
        Node owner;
        const std::string name;
        Operator(std::string name,
                 GraphInPtr graph):
                name(name),
                graph(graph)
        {};

        virtual ad_value_type get_value_type() = 0;
        virtual Shape get_shape() = 0;
        virtual ad_node_type get_node_type() = 0;
        virtual size_t get_gradient_level() = 0;
        virtual NodeVec get_parents() = 0;
        virtual NodeVec get_arguments() = 0;

        NodeVec get_ancestors(){
            auto parents = this->get_parents();
            auto arguments = this->get_arguments();
            for(int i=0; i<arguments.size();i++){
                parents.push_back(arguments[i]);
            }
            return parents;
        }

        virtual Node get_parent_grad(Node my_grad, size_t index) = 0;
        void send_grad_message(Node target, Node msg, std::unordered_map<Node, Node> &messages);

        void generate_gradients(std::unordered_map<Node, Node>& messages) {
            // Check for any incoming messages
            if(messages.find(owner) == messages.end()){
                return;
            }

            // Get the gradient with respect to this node
            Node my_grad = messages[owner];
            // Update the message name
            if(my_grad->name == "Derived Node" or my_grad->name == ""){
                my_grad->name = "Grad of " + std::to_string(owner->id);
            } else {
                my_grad->name += "|Grad of " + std::to_string(owner->id);
            }

            // Check for any surpirses, where all parents are constants
            // If that is the case this node should have been constant as well
            // and no message should have been sent to it
            NodeVec parents = get_parents();
            bool constant = true;
            for(int i=0;i<parents.size();i++){
                if(not parents[i]->is_constant()){
                    constant = false;
                    break;
                }
            }
            if(constant){
                throw UnknownError({parents}, "Gradient message present, all parents are constants");
            }

            // Compute and send gradients only to non constant nodes
            for(size_t i=0;i<parents.size();i++) {
                if(not parents[i]->is_constant()) {
                    Node parent_grad = get_parent_grad(my_grad, i);
                    parent_grad->name =
                            "Grad msg " + std::to_string(owner->id) + " -> " + std::to_string(parents[i]->id);
                    send_grad_message(parents[i], parent_grad, messages);
                }
            }
        };
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
        NodeVec children;
        size_t grad_level;
        af::array value;
        SharedPtr shared;

        ExecutionData execution;

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
                     size_t grad_level):
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

        bool is_constant() const{
            if(type == CONSTANT or type == CONSTANT_DERIVED
                    or type == SYMBOLIC_INTEGER or type == UPDATE){
                return true;
            } else {
                return false;
            }
        }

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

        void update_grad_level(){
            if(id == graph->nodes.size()-1){
                NodeVec parents = op->get_parents();
                for(int i=0;i<parents.size();i++){
                    if(grad_level < parents[i]->grad_level){
                        grad_level = parents[i]->grad_level;
                    }
                }
            }
        }

        template <typename T>
        Node apply();
        Node constant();

        Node gt(Node node);
        Node ge(Node node);
        Node lt(Node node);
        Node le(Node node);
        Node eq(Node node);
        Node neq(Node node);
        Node approx_eq(Node node);
        Node approx_neq(Node node);
        Node logical_and(Node node);
        Node logical_or(Node node);
        Node zero_elem();
        Node non_zero_elem();
        Node is_nan();
        Node is_inf();
        Node select(Node result_true, Node result_false);

        Node broadcast(Shape shape);
        Node broadcast_to(Node other) {
            return broadcast(other->shape);
        }
        Node neg();
        Node div();
        Node sum(std::vector<size_t> axes = {0, 1, 2, 3});
        Node square();

        Node exp();
        Node log();
        Node softplus(double threshold = 50);
        Node abs();
        Node sigmoid();
        Node sin();
        Node cos();
        Node tan();
        Node cot();
        Node sinh();
        Node cosh();
        Node tanh();
        Node coth();
        Node pow(Node power);
        Node pow(double power);

        Node relu();

        Node transpose();
        Node minv();
        Node det();
        Node logdet();
        Node trace();

        Node diag();
        Node reshape(Shape shape);
        Node flatten(size_t ndim = 1);
        Node reorder(std::array<size_t, 4> order);
        Node reorder(size_t dim0, size_t dim1, size_t dim2=2, size_t dim3=3);
    };

    typedef NodeInternal* Node;
    typedef std::vector<Node> NodeVec;
    typedef std::vector<std::pair<Node, Node>> Updates;


    class GraphInternal : public std::enable_shared_from_this<GraphInternal> {
    public:
        std::vector<std::shared_ptr<NodeInternal>> nodes;
        std::string name;
        Device default_device;
        ad_float_type f_type;
        ad_integer_type i_type;
        ad_implicit_broadcast broadcast;
        size_t sym_integer_count;
        std::vector<SharedPtr> shared_vars;

        std::vector<size_t> temporary_constants;
        std::vector<size_t> temprary_updates;

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

        std::vector<bool> get_descendands_mask(std::vector<Node> marked);
        std::vector<bool> get_ancestors_mask(std::vector<Node> marked);

        Graph copy(std::vector<bool> mask);

        std::vector<Node> gradient(Node objective, std::vector<Node> params);

        int find_same_node(std::shared_ptr<Operator> op);

        Graph optimize(std::vector<Node>& targets, Updates& updates);

        Node derived_node(std::shared_ptr<Operator> op,
                          int grad_level = -1);

        Node update_node(Node shared, Node update, int grad_level = -1);

        void update(Node shared, Node update);

        void add_temporary_updates(const Updates& updates);

        void clear_temprary_updates();

        Node constant_node(af::array value);

        Node shared_var(af::array value, std::string name = "SharedVar");

        Node tensor(ad_value_type v_type,
                    std::array<SymInt, 4> shape,
                    std::string name = "InputTensor");

        Node tensor(ad_value_type v_type,
                    SymInt shape0,
                    SymInt shape1,
                    SymInt shape2,
                    SymInt shape3,
                    std::string name = "InputTensor");

        Node tensor(ad_value_type v_type,
                    std::string name = "InputTensor");

        Node tensor_as(Node node,
                       std::string name = "InputTensor");

        Node tensor3(ad_value_type v_type,
                     std::array<SymInt, 3> shape,
                     std::string name = "InputTensor3");

        Node tensor3(ad_value_type v_type,
                     SymInt shape0,
                     SymInt shape1,
                     SymInt shape2,
                     std::string name = "InputTensor3");

        Node tensor3(ad_value_type v_type,
                     std::string name = "InputTensor3");


        Node tensor3_as(Node node,
                        std::string name = "InputTensor3");

        Node matrix(ad_value_type v_type,
                    std::array<SymInt, 2> shape,
                    std::string name = "InputMatrix");

        Node matrix(ad_value_type v_type,
                    SymInt shape0,
                    SymInt shape1,
                    std::string name = "InputMatrix");

        Node matrix(ad_value_type v_type,
                    std::string name = "InputMatrix");


        Node matrix_as(Node node,
                       std::string name = "InputMatrix");

        Node square_matrix(ad_value_type v_type,
                           SymInt shape,
                           std::string name = "InputMatrix");

        Node vector(ad_value_type v_type,
                    SymInt shape,
                    std::string name = "InputVector");

        Node vector(ad_value_type v_type,
                    std::string name = "InputVector");


        Node vector_as(Node node,
                       std::string name = "InputVector");

        Node scalar(ad_value_type v_type,
                    std::string name = "InputScalar");

        Node eye(SymInt size);
        Node ones(Shape shape);
        Node ones_like(Node node){
            return ones(node->shape);
        }
        Node zeros(Shape shape);
        Node zeros_like(Node node){
            return zeros(node->shape);
        }
        Node value(double value, Shape shape = {1, 1, 1, 1});
        Node value_like(double value, Node node){
            return value(value, node->shape);
        }
    };

    template <typename T>
    Node NodeInternal::apply() {
        return graph->derived_node(std::make_shared<T>(graph, this));
    }


    template <typename T>
    Node apply(Node parent1, Node parent2){
        GraphInPtr graph = parent1->graph;
        return graph->derived_node(std::make_shared<T>(graph, parent1, parent2);
    }

    template <typename T>
    Node apply(NodeVec parents){
        GraphInPtr graph = parents[0]->graph;
        return graph->derived_node(std::make_shared<T>(graph, parents));
    }

    class Input : public Operator {
    public:
        Input(GraphInPtr graph):
                Operator("Input", graph){}

        ad_value_type get_value_type(){
            return ad_value_type::FLOAT;
        }

        Shape get_shape(){
            return Shape{0, 0, 0, 0};
        }

        ad_node_type get_node_type(){
            return INPUT;
        };

        size_t get_gradient_level(){
            return 0;
        }

        NodeVec get_parents(){
            return NodeVec {};
        }

        NodeVec get_arguments(){
            return NodeVec {};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }

        void generate_gradients(std::unordered_map<Node, Node>& messages){
            if(messages.find(owner) == messages.end()){
                return;
            }
            if(owner->name == "Derived Node"){
                owner->name = "Grad of " + std::to_string(owner->id);
            } else {
                owner->name += "|Grad of " + std::to_string(owner->id);
            }
        };
    };

    class Update : public Operator {
    public:
        Node shared;
        Node update;
        Update(GraphInPtr graph,
               Node shared,
               Node update):
                Operator("Update", graph),
                shared(shared),
                update(update){
            verify_inputs();
        }

        void verify_inputs();

        ad_value_type get_value_type(){
            return shared->v_type;
        }

        Shape get_shape(){
            return Shape{0, 0, 0, 0};
        }

        ad_node_type get_node_type(){
            return UPDATE;
        };

        size_t get_gradient_level(){
            return update->grad_level;
        }

        NodeVec get_parents(){
            return {update};
        }

        NodeVec get_arguments(){
            return {shared};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }
    };

    std::vector<bool> GraphInternal::get_descendands_mask(std::vector<Node> marked){
        auto n = this->nodes.size();
        std::vector<bool> descendands_mask(n, false);
        for(int i=0;i<marked.size();i++){
            descendands_mask[marked[i]->id] = true;
        }

        // Mark all direct children
        for(int i=0;i<n; i++){
            if(descendands_mask[i]){
                auto children = nodes[i]->children;
                for(int j=0;j<children.size();j++){
                    descendands_mask[children[j]->id] = true;
                }
            }
        }
        return descendands_mask;
    }

    std::vector<bool> GraphInternal::get_ancestors_mask(std::vector<Node> marked){
        // Assumes that computations are ordered
        auto n = this->nodes.size();
        std::vector<bool> ancestors_mask(n, false);
        for(int i=0;i<marked.size();i++){
            ancestors_mask[marked[i]->id] = true;
        }
        for(int i=0;i<temprary_updates.size();i++){
            auto node = nodes[temprary_updates[i]];
            ancestors_mask[node->op->get_parents()[0]->id] = true;
        }

        // Mark all direct ancesotrs
        for(int i=n-1;i >= 0; i--){
            if(ancestors_mask[i]){
                auto ancestors = nodes[i]->op->get_ancestors();
                for(int j=0;j<ancestors.size();j++){
                    ancestors_mask[ancestors[j]->id] = true;
                }
            }
        }

        return ancestors_mask;
    }

    Graph GraphInternal::copy(std::vector<bool> mask){
        Graph new_graph = create_graph();
        new_graph->name = name + "_copy";
        new_graph->default_device = default_device;
        new_graph->f_type = f_type;
        new_graph->i_type = i_type;
        new_graph->broadcast = broadcast;
        new_graph->sym_integer_count = sym_integer_count;
        new_graph->shared_vars = shared_vars;
        size_t mapping[nodes.size()];
        for(int i=0;i<nodes.size();i++){
            mapping[i] = 0;
        }
        for(int i=0;i<this->nodes.size();i++){
            if(mask[i]){
                NodeVec ancestors = nodes[i]->op->get_ancestors();
                NodeVec new_ancestors;
                for(int j=0;j<ancestors.size();j++){
                    new_ancestors.push_back(new_graph->nodes[mapping[ancestors[j]->id]].get());
                }
                // TODO
//                auto new_op = nodes[i]->op.copy(new_ancestors);
                auto new_op = nodes[i]->op;
                auto node = std::make_shared<NodeInternal>(new_graph.get(),
                                                           nodes[i]->device,
                                                           0,
                                                           nodes[i]->name,
                                                           nodes[i]->type,
                                                           nodes[i]->v_type,
                                                           nodes[i]->shape,
                                                           new_op,
                                                           nodes[i]->grad_level);
                node->id = new_graph->nodes.size();
                new_graph->nodes.push_back(node);
            }
        }
        return new_graph;
    }

    std::vector<Node> GraphInternal::gradient(Node objective, std::vector<Node> params) {
        if(not nodes[objective->id]->is_scalar()){
            throw UnsupportedGradient();
        }
        std::unordered_map<Node, Node> grad_messages;

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
        Node unity_grad = value(1.0);
        unity_grad->grad_level = objective->grad_level + ((unsigned short) 1);
        unity_grad->name = "";
        grad_messages[objective->id] = unity_grad;

        // Send all gradient messages
        for (auto i = flow_tree.size(); i > 0; i--) {
            auto node_id = flow_tree[i-1];
            if (grad_messages.find(node_id) != grad_messages.end()) {
                this->nodes[node_id]->op->generate_gradients(grad_messages);
            }
        }

        // Extract the gradients for each parameter
        std::vector<Node> grads;
        for (int i = 0; i < params.size(); i++) {
            grads.push_back(grad_messages[params[i]]);
        }

        // Restore types of other inputs
        temporary_constants.clear();
        return grads;
    }

    int find_same_node(std::shared_ptr<Operator> op){
        return -1;
    }

    Graph GraphInternal::optimize(std::vector<Node>& targets, Updates& updates){
        Graph copy = this->copy(this->get_ancestors_mask(targets));
        return copy;
    }

    Node GraphInternal::derived_node(std::shared_ptr<Operator> op,
                                     int grad_level) {
        int same_node = find_same_node(op);
        if(same_node == -1) {
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
            op->owner = result.get();
            NodeVec ancestors = op->get_ancestors();
            for (int i = 0; i < ancestors.size(); i++) {
                ancestors[i]->children.push_back(result.get());
            }
            return result.get();
        } else {
            return nodes[same_node].get();
        }
    }

    Node GraphInternal::update_node(Node shared,
                                    Node update,
                                    int grad_level) {
        auto op = std::make_shared<Update>(shared_from_this(), shared, update);
        int same_node = find_same_node(op);
        if(same_node == -1) {
            grad_level = grad_level == -1 ? op->get_gradient_level() : grad_level;
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this(),
                    default_device,
                    nodes.size(),
                    "Update Node",
                    op->get_node_type(),
                    op->get_value_type(),
                    op->get_shape(),
                    op,
                    grad_level
            );
            this->nodes.push_back(result);
            op->owner = result.get();
            NodeVec ancestors = op->get_ancestors();
            for (int i = 0; i < ancestors.size(); i++) {
                ancestors[i]->children.push_back(result.get());
            }
            return result.get();
        } else {
            return nodes[same_node].get();
        }
    }

    void GraphInternal::update(Node shared, Node update){
        shared->graph->update_node(shared, update);
    }

    void GraphInternal::add_temporary_updates(const Updates& updates){
        for(int i=0;i<updates.size();i++){
            size_t id = this->nodes.size();
            update_node (updates[i].first, updates[i].second);
            this->temprary_updates.push_back(id);
        }
    }

    void GraphInternal::clear_temprary_updates(){
        for(int i=0;i<temprary_updates.size();i++){
            this->nodes.pop_back();
        }
        this->temprary_updates.clear();
    }

    Node GraphInternal::constant_node(af::array value){
        ad_value_type dtype;
        if(value.type() == af::dtype::b8){
            dtype = BOOLEAN;
        } else if(value.type() == af::dtype::f32
                  or value.type() == af::dtype::f64){
            dtype = FLOAT;
        } else {
            dtype = INTEGER;
        }
        af::dim4 dims = value.dims();
        std::shared_ptr<NodeInternal> result = std::make_shared<NodeInternal>(
                shared_from_this(),
                default_device,
                nodes.size(),
                "Constant Node",
                ad_node_type::CONSTANT,
                dtype,
                Shape {dims[0], dims[1], dims[2], dims[3]},
                std::make_shared<Input>(shared_from_this()),
                0
        );
        result->value = value;
        this->nodes.push_back(result);
        result->op->owner = result.get();
        return result.get();
    }

    Node GraphInternal::shared_var(af::array value, std::string name){
        ad_value_type dtype;
        if(value.type() == af::dtype::b8){
            dtype = BOOLEAN;
        } else if(value.type() == af::dtype::f32
                  or value.type() == af::dtype::f64){
            dtype = FLOAT;
        } else {
            dtype = INTEGER;
        }
        auto dims = value.dims();
        auto result = std::make_shared<NodeInternal>(
                shared_from_this(),
                default_device,
                nodes.size(),
                name,
                ad_node_type::SHARED_INPUT,
                dtype,
                Shape {dims[0], dims[1], dims[2], dims[3]},
                std::make_shared<Input>(shared_from_this()),
                0
        );
        result->shared = std::make_shared<SharedVariable>(result->id, value);
        this->shared_vars.push_back(result->shared);
        this->nodes.push_back(result);
        result->op->owner = result.get();
        return result.get();
    }

    Node GraphInternal::tensor(ad_value_type v_type,
                               std::array<SymInt, 4> shape,
                               std::string name) {
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
        result->op->owner = result.get();
        return result.get();
    }

    Node GraphInternal::tensor(ad_value_type v_type,
                               SymInt shape0,
                               SymInt shape1,
                               SymInt shape2,
                               SymInt shape3,
                               std::string name) {
        std::array<SymInt, 4> shape{shape0,
                                    shape1,
                                    shape2,
                                    shape3};
        return tensor(v_type, shape, name);
    }

    Node GraphInternal::tensor(ad_value_type v_type,
                               std::string name) {
        std::array<SymInt, 4> shape = {
                get_new_symbolic_integer(),
                get_new_symbolic_integer(),
                get_new_symbolic_integer(),
                get_new_symbolic_integer()
        };
        return tensor(v_type, shape, name);
    }

    Node GraphInternal::tensor_as(Node node, std::string name) {
        return tensor(node->v_type, node->shape, name);
    }

    Node GraphInternal::tensor3(ad_value_type v_type,
                                std::array<SymInt, 3> shape,
                                std::string name = "InputTensor3") {
        return tensor(v_type, {shape[0], shape[1], shape[2], 1}, name);
    }

    Node GraphInternal::tensor3(ad_value_type v_type,
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

    Node GraphInternal::tensor3(ad_value_type v_type,
                                std::string name = "InputTensor3") {
        auto shape0 = this->get_new_symbolic_integer();
        auto shape1 = this->get_new_symbolic_integer();
        auto shape2 = this->get_new_symbolic_integer();
        return tensor3(v_type, shape0, shape1, shape2, name);
    }


    Node GraphInternal::tensor3_as(Node node,
                                   std::string name = "InputTensor3") {
        if(not node->is_tensor3()){
            throw "Node with id '" + std::to_string(node->id) + "' is not a tensor3.";
        }
        return tensor3(this->nodes[node->id]->v_type,
                       this->nodes[node->id]->shape[0],
                       this->nodes[node->id]->shape[1],
                       this->nodes[node->id]->shape[2],
                       name);
    }

    Node GraphInternal::matrix(ad_value_type v_type,
                               std::array<SymInt, 2> shape,
                               std::string name = "InputMatrix") {
        std::array<SymInt, 4> shape_t{shape[0], shape[1], SymInt::one(), SymInt::one()};
        return tensor(v_type, shape_t, name);
    }

    Node GraphInternal::matrix(ad_value_type v_type,
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

    Node GraphInternal::matrix(ad_value_type v_type,
                               std::string name = "InputMatrix") {
        auto shape0 = this->get_new_symbolic_integer();
        auto shape1 = this->get_new_symbolic_integer();
        return matrix(v_type, shape0, shape1, name);
    }


    Node GraphInternal::matrix_as(Node node,
                                  std::string name = "InputMatrix") {
        if(not node->is_matrix()){
            throw "Node with id '" + std::to_string(node->id) + "' is not a matrix.";
        }
        return matrix(this->nodes[node->id]->v_type,
                      this->nodes[node->id]->shape[0],
                      this->nodes[node->id]->shape[1],
                      name);
    }

    Node GraphInternal::square_matrix(ad_value_type v_type,
                                      SymInt shape,
                                      std::string name = "InputMatrix") {
        std::array<SymInt, 4> shape_t{shape, shape, SymInt::one(), SymInt::one()};
        return tensor(v_type, shape_t, name);
    }

    Node GraphInternal::vector(ad_value_type v_type,
                               SymInt shape,
                               std::string name = "InputVector") {
        std::array<SymInt, 4> shape_t{shape, SymInt::one(), SymInt::one(), SymInt::one()};
        return tensor(v_type, shape_t, name);
    }

    Node GraphInternal::vector(ad_value_type v_type,
                               std::string name = "InputVector") {
        auto shape0 = this->get_new_symbolic_integer();
        return vector(v_type, shape0, name);
    }


    Node GraphInternal::vector_as(Node node,
                                  std::string name = "InputVector") {
        if(not node->is_vector()){
            throw "Node with id '" + std::to_string(node->id) + "' is not a vector.";
        }
        return vector(this->nodes[node->id]->v_type,
                      this->nodes[node->id]->shape[0],
                      name);
    }

    Node GraphInternal::scalar(ad_value_type v_type,
                               std::string name = "InputScalar") {
        std::array<SymInt, 4> shape_t{SymInt::one(), SymInt::one(), SymInt::one(), SymInt::one()};
        return tensor(v_type, shape_t, name);
    }

    class UnsupportedGradient : public std::exception {
        const char* what() const throw(){
            return "\nThe gradient operation supports only scalar values";
        }
    };

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
                      NodeVec inputs):
                name(name)
        {
            for(int i=0;i < inputs.size(); i++){
                input_ids.push_back(inputs[i]->id);
                input_shapes.push_back(inputs[i]->shape);
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
                           NodeVec inputs) :
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
                          NodeVec inputs) :
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
                         NodeVec inputs,
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

        UnknownError(NodeVec inputs,
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

    void Update::verify_inputs(){
        if(shared.lock()->type != SHARED_INPUT){
            throw InvalidArguments(name, {shared, update},
                                   "First argument should be a shared variable not an expression.");
        }
        auto shared_shape = shared.lock()->shape;
        auto update_shape = update.lock()->shape;
        for(int i=0;i<4;i++){
            if(shared_shape[i] != update_shape[i]){
                throw IncompatibleShapes(name, {shared, update});
            }
        }
        if(shared.lock()->v_type != update.lock()->v_type){
            throw InvalidArguments(name, {shared, update},
                                   "Shared variable and update should have same value type");
        }
    }

    void Update::generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages){
        auto graph = this->graph.lock();
        if (messages.find(current) != messages.end()) {
            throw UnknownError({shared, update},
                               "An update operator recieved a gradient message.");
        }
        return;
    }

    std::string to_string(ad_node_type const & type){
        switch(type){
            case ad_node_type::SYMBOLIC_INTEGER: return "SYMBOLIC_INTEGER";
            case ad_node_type::CONSTANT: return "CONSTANT";
            case ad_node_type::INPUT : return "INPUT";
            case ad_node_type::SHARED_INPUT : return "SHARED";
            case ad_node_type::INPUT_DERIVED: return "DERIVED";
            case ad_node_type::CONSTANT_DERIVED: return "CONSTANT_DERIVED";
            case ad_node_type::UPDATE: return "UPDATE";

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