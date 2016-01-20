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
    const size_t GRAD_LEVEL_BAR = 100;

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
    class NodeInternal;

    class Node {
    public:
        NodeInternal *ptr;

        Node():
                ptr(NULL){};

        Node(const std::shared_ptr<NodeInternal> shared_ptr):
                ptr(shared_ptr.get()) {};

        Node(NodeInternal* node):
            ptr(node) {};

        Node(const Node& node):
                ptr(node.ptr) {};

        Node(const Node* const node):
            ptr(node->ptr) {};

        bool empty(){
            return ptr == NULL;
        }
        void copy_to(GraphInPtr graph, std::vector<Node> ancestors);
        void update(Node update);

        bool is_constant() const;
        bool is_scalar() const;
        bool is_vector() const;
        bool is_vector_strict() const;
        bool is_matrix() const;
        bool is_matrix_strict() const;
        bool is_tensor3() const;
        bool is_tensor3_strict() const;
        bool is_tensor4_strict() const;

//        bool is_constant() const{
//            if(ptr->type == CONSTANT or ptr->type == CONSTANT_DERIVED
//               or ptr->type == SYMBOLIC_INTEGER or ptr->type == UPDATE){
//                return true;
//            } else {
//                return false;
//            }
//        }
//
//        bool is_scalar() const{
//            for(int i=0; i < 4; i++){
//                if(ptr->shape[i] != 1){
//                    return false;
//                }
//            }
//            return true;
//        }
//
//        bool is_vector() const{
//            for(int i=1; i < 4; i++){
//                if(ptr->shape[i] != 1){
//                    return false;
//                }
//            }
//            return true;
//        }
//
//        bool is_vector_strict() const{
//            for(int i=0; i < 1; i++){
//                if(ptr->shape[i] == 1){
//                    return false;
//                }
//            }
//            for(int i=1; i < 4; i++){
//                if(ptr->shape[i] != 1){
//                    return false;
//                }
//            }
//            return true;
//        }
//
//        bool is_matrix() const{
//            for(int i=2; i < 4; i++){
//                if(ptr->shape[i] != 1){
//                    return false;
//                }
//            }
//            return true;
//        }
//
//        bool is_matrix_strict() const{
//            for(int i=0; i < 2; i++){
//                if(ptr->shape[i] == 1){
//                    return false;
//                }
//            }
//            for(int i=2; i < 4; i++){
//                if(ptr->shape[i] != 1){
//                    return false;
//                }
//            }
//            return true;
//        }
//
//        bool is_tensor3() const{
//            for(int i=3; i < 4; i++){
//                if(ptr->shape[i] != 1){
//                    return false;
//                }
//            }
//            return true;
//        }
//
//        bool is_tensor3_strict() const{
//            for(int i=0; i < 3; i++){
//                if(ptr->shape[i] == 1){
//                    return false;
//                }
//            }
//            for(int i=3; i < 4; i++){
//                if(ptr->shape[i] != 1){
//                    return false;
//                }
//            }
//            return true;
//        }
//
//        bool is_tensor4_strict() const{
//            for(int i=0; i < 4; i++){
//                if(ptr->shape[i] == 1){
//                    return false;
//                }
//            }
//            return true;
//        }


//        template <typename T>
//        Node apply();
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

    class Device{
    public:
        ad_device_type type;
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

    class ExecutionData{
    public:
        bool inlined;
        size_t register_id;
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

    class Operator{
    public:
        GraphInPtr graph;
        Node owner;
        const std::string name;
        Operator(std::string name,
                 GraphInPtr graph):
                name(name),
                graph(graph){};

        virtual std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) = 0;
        virtual ad_value_type get_value_type() = 0;
        virtual Shape get_shape() = 0;
        virtual ad_node_type get_node_type() = 0;
        virtual size_t get_gradient_level() = 0;
        virtual NodeVec get_parents() = 0;
        virtual NodeVec get_arguments() = 0;
        virtual Node get_parent_grad(Node my_grad, size_t index) = 0;
        void send_grad_message(size_t target, Node msg, std::vector<Node>& messages);
        virtual void generate_gradients(std::vector<Node>& messages);
        double get_scalar_value();

        NodeVec get_ancestors(){
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
        NodeVec children;
        size_t grad_level;
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
        NodeVec copy(GraphInPtr new_graph, std::vector<bool> mask);
        SymInt get_new_symbolic_integer() {
            this->sym_integer_count++;
            return SymInt::variable(this->sym_integer_count - 1);
        }

        std::vector<bool> get_descendants_mask(std::vector<Node>& marked);
        std::vector<bool> get_ancestors_mask(std::vector<Node>& marked);
        Node find_same_node(std::shared_ptr<Operator> op);
        void add_temporary_updates(const Updates& updates);
        void clear_temporary_updates();
        std::vector<Node> gradient(Node objective, std::vector<Node> params);
        Graph optimize(NodeVec& targets, Updates& updates,NodeVec& inputs,
                       NodeVec& new_targets, Updates& new_updates, NodeVec& new_inputs);

        Node shared_var(af::array value, std::string name = "SharedVar");
        Node derived_node(std::shared_ptr<Operator> op, size_t grad_level = GRAD_LEVEL_BAR);
        Node update_node(Node shared, Node update, size_t grad_level = GRAD_LEVEL_BAR);
        Node constant_node(af::array value);

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
                input_ids.push_back(inputs[i].ptr->id);
                input_shapes.push_back(inputs[i].ptr->shape);
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

    class Input : public Operator {
    public:
        Input(GraphInPtr graph):
                Operator("Input", graph){}

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<Input>(graph);
        }

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
    };

    class Update : public Operator {
    public:
        Node shared;
        Node update;

        void verify_inputs(){
            if(shared.ptr->type != SHARED_INPUT){
                throw InvalidArguments(name, {shared, update},
                                       "First argument should be a shared variable not an expression.");
            }
            auto shared_shape = shared.ptr->shape;
            auto update_shape = update.ptr->shape;
            for(int i=0;i<4;i++){
                if(shared_shape[i] != update_shape[i]){
                    throw IncompatibleShapes(name, {shared, update});
                }
            }
            if(shared.ptr->v_type != update.ptr->v_type){
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

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<Update>(graph, ancestors[0], ancestors[1]);
        }

        ad_value_type get_value_type(){
            return shared.ptr->v_type;
        }

        Shape get_shape(){
            return Shape{0, 0, 0, 0};
        }

        ad_node_type get_node_type(){
            return UPDATE;
        };

        size_t get_gradient_level(){
            return update.ptr->grad_level;
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