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

    enum ad_node_type{
        /**
         * The node is just a SymInt, which interacts with other nodes in an operator
         */
        SYMBOLIC_INTEGER,
        /**
         * The node is a constant
         */
        CONSTANT,
        /**
         * The node is derived from a constant, trough some operator manipulations
         */
        CONSTANT_DERIVED,
        /**
         * The node is an input
         */
        INPUT,
        /**
         * The node is a shared variable
         */
        SHARED_INPUT,
        /**
         * The node is derived from at least one input
         */
        INPUT_DERIVED,
        /**
         * The node is an update to shared variable
         */
        UPDATE
    };

    enum ad_value_type{
        /**
         * Represents floating point values, inrespectable of precision
         */
        FLOAT,
        /**
         * Represents integer values, inrespectable of precision
         */
        INTEGER,
        /**
         * Represents boolean values
         */
        BOOLEAN
    };

    enum ad_device_type {
        /**
         * The device is one or more CPUs
         */
        CPU,
        /**
         * The device is a single GPU
         */
        GPU
    };

    enum ad_implicit_broadcast {
        /**
         * If any node performs an implicit broadcast an exception is thrown
         */
        RAISE,
        /**
         * If any node performs an implicit broadcast a warning is printed to the standard output
         */
        WARN,
        /**
         * If any node performs an implicit broadcast there is no notification
         */
        QUIET
    };

    enum ad_float_type {
        /**
         * 16 bit floating point number
         */
        f16,
        /**
         * 32 bit floating point number
         */
        f32,
        /**
         * 64 bit floating point number
         */
        f64,
    };
    enum ad_integer_type {
        /**
         * Signed 8 bit integer
         */
        s8,
        /**
         * Unsigned 8 bit integer
         */
        u8,
        /**
        * Signed 16 bit integer
        */
        s16,
        /**
         * Unsigned 16 bit integer
         */
        u16,
        /**
        * Signed 32 bit integer
        */
        s32,
        /**
         * Unsigned 32 bit integer
         */
        u32,
        /**
        * Signed 64 bit integer
        */
        s64,
        /**
         * Unsigned 64 bit integer
         */
        u64
    };

    class GraphInternal;
    typedef GraphInternal* GraphInPtr;
    typedef std::shared_ptr<GraphInternal> Graph;
    class NodeInternal;

    /**
     * The class is an API wrapper around node of the graph
     */
    class Node {
    public:
        std::weak_ptr<NodeInternal> ptr;

        Node(){};

        /**
         * Unwraps the pointer to the internal node. Exits the program if the pointer has expired
         */
        std::shared_ptr<NodeInternal> unwrap() const{
            if(ptr.expired()){
                std::cerr << "Trying to access a Node whose pointer has expired" << std::endl;
                exit(1);
            }
            return ptr.lock();
        }

        Node(const std::shared_ptr<NodeInternal> shared_ptr):
                ptr(shared_ptr) {};

        Node(const Node& node):
                ptr(node.ptr) {};

        Node(const Node* node):
                ptr(node->ptr) {};

        /**
         * Checks if the Node is empty, could happen only if you manually construct this class
         * or the Graph instance is out of scope.
         */
        bool empty(){
            return ptr.expired();
        }

        /**
         * Copies the node to another graph, by using the ancestors provided
         */
        void copy_to(GraphInPtr graph, std::vector<Node> ancestors);

        bool is_constant() const;
        bool is_scalar() const;
        bool is_vector() const;
        bool is_vector_strict() const;
        bool is_matrix() const;
        bool is_matrix_strict() const;
        bool is_tensor3() const;
        bool is_tensor3_strict() const;
        bool is_tensor4_strict() const;

        void update(Node update);
        Node alias();
        Node broadcast(Shape shape);
        Node broadcast_to(Node other);

        Node neg();
        Node div();
        Node sum(std::vector<size_t> axes = {0, 1, 2, 3});
        Node square();

        Node constant();
        Node gt(Node node);
        Node ge(Node node);
        Node lt(Node node);
        Node le(Node node);
        Node eq(Node node);
        Node neq(Node node);
        Node approx_eq(Node node, double tol=0.00001);
        Node approx_neq(Node node, double tol=0.00001);
        Node logical_and(Node node);
        Node logical_or(Node node);
        Node zero_elem();
        Node non_zero_elem();
        Node is_nan();
        Node is_inf();
        Node select(Node result_true, Node result_false);

        Node exp();
        Node log();
        Node log1p();
        Node softplus(size_t threshold = 50);
        Node abs();
        Node sigmoid();
        Node relu();
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
    typedef std::vector<Node> NodeVec;
    typedef std::vector<std::pair<Node, Node>> Updates;

    /**
     * A single computational device to faciliate mulit node computations
     */
    class Device{
    public:
        /**
         * Type of the device
         */
        ad_device_type type;
        /**
         * Id of the device
         */
        size_t id;
        Device():
                type(ad_device_type::CPU),
                id(0) {};

        Device(const ad_device_type type, const size_t id):
                type(type),
                id(id) {};

        Device(Device& device):
                type(device.type),
                id(device.id) {};
    };

    /**
     * The class provides data generated by the graph optimizer relevant to the
     * backends which generate code
     */
    class ExecutionData{
    public:
        /**
         * Whether the node should be inlined
         */
        bool inlined;
        /**
         * The graph optimizer register allocation
         */
        size_t register_id;
        /**
         * For synchronization and memory allocation this will contain the time step after
         * which the node can be destroyed
         */
        size_t lifespan;
        ExecutionData():
                inlined(false),
                register_id(0),
                lifespan(0) {};

        ExecutionData(const ExecutionData& data):
                inlined(data.inlined),
                register_id(data.register_id),
                lifespan(data.lifespan) {};
    };

    /**
     * Abstract class for operators
     */
    class Operator{
    public:
        /**
         * Pointer to the owning graph
         */
        GraphInPtr graph;
        /**
         * Pointer to the owning node
         */
        Node owner;
        /**
         * Unique name of the concrete operator classes
         */
        const std::string name;
        Operator(std::string name,
                 GraphInPtr graph):
                name(name),
                graph(graph){};

        /**
         * Copies the operator to a new graph
         */
        virtual std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const = 0;
        /**
         * Gives the value type of the result node of the operator
         */
        virtual ad_value_type get_value_type() const = 0;

        /**
         * Gives the shape of the result node of the operator
         */
        virtual Shape get_shape() const = 0;
        /**
         * Gives the node type of the result node of the operator
         */
        virtual ad_node_type get_node_type() const = 0;
        /**
         * Gives the gradient level of the node result of the operator
         */
        virtual size_t get_gradient_level() const = 0;
        /**
         * Gives the parents of the node result of the operator
         */
        virtual NodeVec get_parents() const = 0;
        /**
         * Gives the arguments of the node result of the operator
         */
        virtual NodeVec get_arguments() const = 0;
        /**
         * A function which shoulde compute and return the gradient with respect
         * to the parent and the specified index, given the gradient of the owner node
         */
        virtual Node get_parent_grad(Node my_grad, size_t index) = 0;
        /**
         * Sends a gradient message to the target by either inserting it in the messages
         * if none exists, or accumulating them
         */
        void send_grad_message(size_t target, Node msg, std::vector<Node>& messages) const;
        /**
         * Generates all gradients, given the current setup of the graph
         */
        virtual void generate_gradients(std::vector<Node>& messages);
        /**
         * Returns a scalar value if the operator is constant
         */
        double get_scalar_value() const;
        /**
         * Compares only if this operator is equal to the other, not the other way around.
         * Note that although equality is symmetric, because of mathematical idenitities
         * and the fact that the code is with each operator separately the true equality
         * operator is `op1.equals(op2) or op2.equals(op1)`
         */
        virtual bool equals(const std::shared_ptr<Operator> op) const = 0;
        /**
         * Returns the union of the parents and arguments of the node result of the operator
         */
        NodeVec get_ancestors() const {
            auto parents = this->get_parents();
            auto arguments = this->get_arguments();
            for(int i=0; i<arguments.size();i++){
                parents.push_back(arguments[i]);
            }
            return parents;
        }
    };

    /**
     * Internal class for a graph node
     */
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
        SymInt sym_value;
        af::array value;
        SharedPtr shared;

        ExecutionData execution;

        NodeInternal(GraphInPtr graph, Device device):
                graph(graph),
                device(device) {}

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
                shape(shape) {}
    };

    /**
     * Internal class for a compute graph
     */
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
        size_t gradient_mode;

        std::vector<Node> temporary_constants;
        std::vector<Node> temporary_updates;

        GraphInternal() {
            // TODO Have a better preference of devices available in order
            name = "Function";
            sym_integer_count = 0;
            default_device = Device(CPU, 0);
            f_type = ad_float_type::f32;
            i_type = ad_integer_type::s32;
            broadcast = ad_implicit_broadcast::RAISE;
            gradient_mode = 0;
        }

        /**
         * Copies all nodes for which the mask is true
         */
        NodeVec copy(GraphInPtr new_graph, std::vector<bool> mask);
        /**
         * Returns an array masking all descendats of the marked nodes
         */
        std::vector<bool> get_descendants_mask(std::vector<Node>& marked);
        /**
         * Returns an array masking all ancestors of the marked nodes
         */
        std::vector<bool> get_ancestors_mask(std::vector<Node>& marked);
        Node find_same_node(std::shared_ptr<Operator> op);
        /**
         * Adds temporary updates to the graph
         */
        void add_temporary_updates(const Updates& updates);
        /**
         * Removes all temporary updates of the graph
         */
        void clear_temporary_updates();
        /**
         * Returns the gradients of the objective with respect to the parameters provided
         */
        std::vector<Node> gradient(Node objective, std::vector<Node> params);
        /**
         * Optimizes a graph with respect to the given nodes (INTERNAL)
         */
        Graph optimize(NodeVec& targets, Updates& updates,NodeVec& inputs,
                       NodeVec& new_targets, Updates& new_updates, NodeVec& new_inputs);

        /**
         * Creates a new shared variable
         */
        Node shared_var(af::array value, std::string name = "SharedVar");
        /**
         * Creates a new derived node (INTERNAL)
         */
        Node derived_node(std::shared_ptr<Operator> op);
        /**
         * Creates a new update node
         */
        Node update_node(Node shared, Node update);
        /**
         * Creates a new constant node
         */
        Node constant_node(af::array value);

        /**
         * Returns the next unused symbolic integer
         */
        SymInt get_new_symbolic_integer() {
            this->sym_integer_count++;
            return SymInt::variable(this->sym_integer_count - 1);
        }

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
        Node zeros(Shape shape);
        Node constant_value(double value, Shape shape = {1, 1, 1, 1});
    };

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
                name(name) {
            for(int i=0;i < inputs.size(); i++){
                input_ids.push_back(inputs[i].unwrap()->id);
                input_shapes.push_back(inputs[i].unwrap()->shape);
            }
        };

        virtual std::string get_message() const = 0;
        const char* what() const throw(){
            return this->get_message().c_str();
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

    /**
     * Skips any alias operators to get the base operator
     */
    std::shared_ptr<Operator> get_base_op(const std::shared_ptr<Operator> op){
        std::shared_ptr<Operator> base_op = op;
        while(op->name == "Alias"){
            base_op = base_op->get_parents()[0].unwrap()->op;
        }
        return op;
    }

//    /**
//     * Compares if the two operators are symboliclly equivalent.
//     * Note this does not make any nodes on the graph and is not related to computation.
//     */
//    bool symbolic_equals(const std::shared_ptr<Operator> op1,
//                         const std::shared_ptr<Operator> op2){
//        return bas_op1->symbolic_equals(bas_op2) or bas_op2->symbolic_equals(bas_op1);
//    }

    /**
     * Check if two nodes are symbolically equivalent.
     */
    bool symbolic_equals(const Node& node1, const Node& node2){
        if(node1.unwrap()->id == node2.unwrap()->id){
            return true;
        }
        if(node1.unwrap()->type != node2.unwrap()->type){
            return false;
        }
        ad_node_type type = node1.unwrap()->type;
        switch (type){
            case SYMBOLIC_INTEGER:{
                return node1.unwrap()->sym_value == node2.unwrap()->sym_value;
            }
            case INPUT: {
                return false;
            }
            case SHARED_INPUT: {
                return false;
            }
            default: {
                const std::shared_ptr<Operator> base_op1 = get_base_op(node1.unwrap()->op);
                const std::shared_ptr<Operator> base_op2 = get_base_op(node2.unwrap()->op);
                return base_op1->equals(base_op2) or base_op2->equals(base_op1);
            }
        }
        return false;
    };

    /**
     * Operator for any inputs
     */
    class Input : public Operator {
    public:
        Input(GraphInPtr graph):
                Operator("Input", graph){}

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors)  const{
            return std::make_shared<Input>(graph);
        }

        ad_value_type get_value_type()  const{
            return ad_value_type::FLOAT;
        }

        Shape get_shape()  const{
            return Shape{0, 0, 0, 0};
        }

        ad_node_type get_node_type()  const{
            return INPUT;
        };

        size_t get_gradient_level()  const{
            return 0;
        }

        NodeVec get_parents()  const{
            return NodeVec {};
        }

        NodeVec get_arguments()  const{
            return NodeVec {};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }

        bool equals(const std::shared_ptr<Operator> op)  const{
            return false;
        }
    };

    /**
     * Operator for updates
     */
    class Update : public Operator {
    public:
        Node shared;
        Node update;

        void verify_inputs(){
            if(shared.unwrap()->type != SHARED_INPUT){
                throw InvalidArguments(name, {shared, update},
                                       "First argument should be a shared variable not an expression.");
            }
            auto shared_shape = shared.unwrap()->shape;
            auto update_shape = update.unwrap()->shape;
            for(int i=0;i<4;i++){
                if(shared_shape[i] != update_shape[i]){
                    throw IncompatibleShapes(name, {shared, update});
                }
            }
            if(shared.unwrap()->v_type != update.unwrap()->v_type){
                throw InvalidArguments(name, {shared, update},
                                       "Shared variable and update should have same value type");
            }
        }

        Update(GraphInPtr graph,
               Node shared,
               Node update):
                Operator("Update", graph),
                shared(shared),
                update(update){
            verify_inputs();
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Update>(graph, ancestors[0], ancestors[1]);
        }

        ad_value_type get_value_type() const{
            return shared.unwrap()->v_type;
        }

        Shape get_shape() const{
            return Shape{0, 0, 0, 0};
        }

        ad_node_type get_node_type() const{
            return UPDATE;
        };

        size_t get_gradient_level() const{
            return update.unwrap()->grad_level;
        }

        NodeVec get_parents() const{
            return {update};
        }

        NodeVec get_arguments() const{
            return {shared};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            return false;
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
//            case ad_float_type::c16: return "c16";
            case ad_float_type::f32: return "f32";
//            case ad_float_type::c32: return "c32";
            case ad_float_type::f64: return "f64";
//            case ad_float_type::c64: return "c64";
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