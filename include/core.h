//
// Created by alex on 10/12/15.
//

#ifndef METADIFF_CORE_H
#define METADIFF_CORE_H

namespace metadiff {

    namespace core {
        /** The maximum number of symbolic integers allowed */
        static size_t const N = 1000;

        /** The root NodeGroup name */
        static std::string const GROUP_ROOT  = "_root";

        /** The NodeGroup name separator */
        static char const GROUP_DELIMITER = '/';

        /**
         * When calling an Operator working along one axis
         * this flag for the axis indicates to auto infer it.
         * (Can we make this int8?)
         */
        static short const AUTO_INFER_AXIS = 1000;

        /** Each Node on the Graph is exactly one of these types */
        enum nodeType {
            /**
    //         * The node is just a SymInt, which interacts with other nodes in an operator
    //         */
//                SYMBOLIC_INTEGER,
            /** The node represents a constant */
                    CONSTANT = 0,
            /** The node is derived from a constant, trough one or more Operator */
                    CONSTANT_DERIVED = 1,
            /**
             * The node is an input.
             * This can be either function input or a shared variable
             */
                    INPUT = 2,
//        /**
//         * The node is a shared variable
//         */
//                SHARED_INPUT,
            /** The node is derived from an input, trough one or more Operator */
                    INPUT_DERIVED = 3
        };

        /**
         * Data type of a Node
         *
         * Note that currently not all provided types are supported
         */
        enum dType{
            /** 8 bit boolean */
                    b8 = 0,
            /** 8 bit unsigned integer */
                    u8 = 1,
            /** 16 bit unsigned integer */
                    u16 = 2,
            /** 32 bit unsigned integer */
                    u32 = 3,
            /** 64 bit unsigned integer */
                    u64 = 4,
            /** 8 bit signed integer */
                    i8 = 5,
            /** 16 bit signed integer */
                    i16 = 6,
            /** 32 bit signed integer */
                    i32 = 7,
            /** 64 bit signed integer */
                    i64 = 8,
            /** 8 bit floating point */
                    f8 = 9,
            /** 16 bit floating point */
                    f16 = 10,
            /** 32 bit floating point */
                    f32 = 11,
            /** 64 bit floating point */
                    f64 = 12
        };


        /** Converts an af_dtype to dType
         * TODO: Make proper exception when given complex type */
        dType convert_af_dtype(af_dtype dtype);

        /**
         * This is the default data type promotion,
         * executed when two nodes of different #dType are used by an Operator.
         * If any of the two types is floating point, the result will be a floating point.
         * Else If any of the two types is integer, the result will be an integer.
         * Else the result is a boolean.
         * The exact precision is determined by the highest precision of the two operands
         * and the maximum allowed for the resulting type.
         */
        dType default_dType_promotion(dType type1,
                                      dType type2,
                                      dType max_float,
                                      dType max_int);

//        enum numeric_type {
//            /**
//             * Represents floating point values
//             */
//                    FLOAT,
//            /**
//             * Represents integer values
//             */
//                    SIGNED_INTEGER,
//            /**
//             * Represents integer values
//             */
//                    UNSIGNED_INTEGER,
//            /**
//             * Represents boolean values
//             */
//                    BOOLEAN
//        };
//
//        enum bit_size {
//            /**
//             * 8 bit numeric
//             */
//                    bit8,
//            /**
//             * 16 bit numeric
//             */
//                    bit16,
//            /**
//             * 32 bit numeric
//             */
//                    bit32,
//            /**
//             * 64 bit numeric
//             */
//                    bit64
//        };

        /**
         * An error policy defines how should we behave when an error occurs
         */
        enum errorPolicy {
            /** Does nothing */
                    QUIET = 0,
            /** Prints a warning */
                    WARN = 1,
            /** Throws an error */
                    RAISE = 2
        };

        /** Given an error executes the errorPolicy */
        void operate_policy(errorPolicy policy,
                            std::shared_ptr<spdlog::logger> const logger,
                            std::exception const & exception);

        /**
         * Currently we support only two device types
         */
        enum deviceType {
            /** Represents a host with one or more CPUs */
                    HOST = 0,
            /** Represents a single GPU */
                    GPU = 1
        };

        /**
         * A single computational device to facilitate multy node computations
         * TODO not yet well designed, high probability it will change in the future
         */
        class Device {
        public:
            /** Type of the device - host or gpu*/
            deviceType type;
            /** A unique identifier of the device */
            size_t id;

            Device() :
                    type(HOST),
                    id(0) { };

            Device(deviceType type, size_t id) :
                    type(type),
                    id(id) { };

            Device(Device &device) :
                    type(device.type),
                    id(device.id) { };
        };


//        class ValueType {
//        public:
//            numeric_type num;
//            bit_size size;
//
//            ValueType(numeric_type num, bit_size size) :
//                    num(num), size(size) { };
//
//            ValueType() : ValueType(FLOAT, bit32) { };
//        };
//
//        static const ValueType f64 = ValueType(FLOAT, bit64);
//        static const ValueType f32 = ValueType(FLOAT, bit32);
//        static const ValueType f16 = ValueType(FLOAT, bit16);
//        static const ValueType f8 = ValueType(FLOAT, bit8);
//        static const ValueType i64 = ValueType(SIGNED_INTEGER, bit64);
//        static const ValueType i32 = ValueType(SIGNED_INTEGER, bit32);
//        static const ValueType i16 = ValueType(SIGNED_INTEGER, bit16);
//        static const ValueType i8 = ValueType(SIGNED_INTEGER, bit8);
//        static const ValueType u64 = ValueType(UNSIGNED_INTEGER, bit64);
//        static const ValueType u32 = ValueType(UNSIGNED_INTEGER, bit32);
//        static const ValueType u16 = ValueType(UNSIGNED_INTEGER, bit16);
//        static const ValueType u8 = ValueType(UNSIGNED_INTEGER, bit8);
//        static const ValueType b8 = ValueType(BOOLEAN, bit8);
//        bool operator==(ValueType const &type1, ValueType const &type2);

//    enum ad_float_type {
//        /**
//         * 16 bit floating point number
//         */
//                f16,
//        /**
//         * 32 bit floating point number
//         */
//                f32,
//        /**
//         * 64 bit floating point number
//         */
//                f64,
//    };
//    enum ad_integer_type {
//        /**
//         * 8 bit integer
//         */
//                i8,
//        /**
//        * 16 bit integer
//        */
//                i16,
//        /**
//        * 32 bit integer
//        */
//                i32,
//        /**
//        * 64 bit integer
//        */
//                i64,
//    };

        /**
         * The class provides data generated by the graph optimizer relevant to the backends
         * TODO This is not yet complete, high probability it will expand in the future
         */
        class ExecutionData {
        public:
            /** Whether the computation should be inlined */
            bool inlined;
            /**
             * Whether the node should be computed in place
             * This is possible only when some of the operands lifespan expires
             */
            bool inplace;
            /** The graph optimizer allocated register id*/
            size_t register_id;
            /**
             * For synchronization and memory allocation this will contain the time step after
             * which the node can be destroyed
             */
            size_t lifespan;

            ExecutionData() :
                    inlined(false),
                    inplace(false),
                    register_id(0),
                    lifespan(0) { };

            ExecutionData(ExecutionData const &data) :
                    inlined(data.inlined),
                    register_id(data.register_id),
                    lifespan(data.lifespan) { };
        };

        /**
         * A NodeGroup is an abstraction of grouping together nodes (and groups).
         * How they are grouped is fully determinate by the user.
         * The hierarchy of groups is necessarily a DAG as well starting with a single root group.
         * The main goal of the groups is to provide a better way of visualizing the computation,
         * as well as a block for naming parameters accordingly.
         */
        class NodeGroup {
        public:
            /** The name of this group */
            std::string const name;
            /** This is the full name of the group, which depends on the parent */
            std::string full_name;
            /** The parent NodeGroup */
            std::weak_ptr<NodeGroup> const parent;
            /** The children groups */
            std::vector<std::weak_ptr<NodeGroup>> children;

            NodeGroup() :
                    name(GROUP_ROOT),
                    full_name(GROUP_ROOT) { };

            NodeGroup(std::string name,
                      std::weak_ptr<NodeGroup> parent) :
                    name(name),
                    parent(parent) {
                if (parent.lock()->full_name == GROUP_ROOT) {
                    full_name = name;
                } else {
                    full_name = parent.lock()->full_name;
                    full_name += GROUP_DELIMITER;
                    full_name += name;
                }
            };

            /**
             * Returns the full name of the group, traversing all its ancestors
             */
//            std::string get_full_name() const {
//                return full_name;
//            std::string full_name = name;
//            NodeGroup* node = parent.lock().get();
//            while(node->name != "_root"){
//                full_name.insert(0, 1, delimiter);
//                full_name.insert(0, node->name);
//                node = node->parent.lock().get();
//            }
//            return full_name;
//            }
        };

        // A few forward declarations and typdefs, unfortunately needed

        /** Axes are used for certain operators */
        typedef std::vector<short> Axes;
        /** A symbolic integer is just a SymbolicPolynomial */
        typedef symbolic::SymbolicPolynomial<N, unsigned short> SymInt;
        /**
        * The shape of any variable.
        * Currently we support 4 dimensional tensors.
        * Each dimension is a SymInt
        */
        typedef std::array<SymInt, 4> Shape;
        /** A group is a weak_ptr to internal Group */
        typedef std::weak_ptr<core::NodeGroup> Group;
        class GraphInternal;
        class NodeInternal;
        class Node;
        /** Vector of Nodes */
        typedef std::vector<Node> NodeVec;
        /** An update is a pair of shared variable and a node */
        typedef std::vector<std::pair<Node, Node>> Updates;
        /** Just a pointer to GraphInternal */
        typedef GraphInternal* GraphInPtr;
        /** A shared_ptr to GraphInternal, this is the outside API */
        typedef std::shared_ptr<core::GraphInternal> Graph;

        // Helper function to get all elements of a node
        SymInt number_of_elements(Shape shape) {
            return shape[0] * shape[1] * shape[2] * shape[3];
        };

        /** The class is an API wrapper around a NodeInternal */
        class Node {
        private:
            std::shared_ptr<spdlog::logger> logger() const {
                return metadiff::logger("node");
            }

        public:
            std::weak_ptr<NodeInternal> ptr;

            Node() {};

            Node(std::shared_ptr<NodeInternal> const ptr) :
                    ptr(ptr) { };

            Node(Node const &node) :
                    ptr(node.ptr) { };

            Node(Node const *node) :
                    ptr(node->ptr) { };

            /** Unwraps the pointer to the internal node.
             * Exits the program if the pointer has expired */
            std::shared_ptr<NodeInternal> unwrap() const;

            /**
             * This operator is overloaded to call class: unwrap()
             */
            std::shared_ptr<NodeInternal> operator->() const;

            /** Copies the node to another graph, by using the ancestors
             * provided from the new graph */
            void copy_to(const GraphInPtr graph, NodeVec ancestors) const;

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

            Node cast(dType dtype);

            Node broadcast(Shape shape);

            Node broadcast_to(Node other);

            static Node add(NodeVec nodes);

            static Node add(Node node1, Node node2);

            Node neg();

            static Node mul(NodeVec nodes);

            static Node mul(Node node1, Node node2);

            Node div();

            Node sum(Axes axes = {0, 1, 2, 3});

            Node square();

            Node as_constant();

            Node logical_not();

            Node logical_and(Node node);

            Node logical_or(Node node);

            Node gt(Node node);

            Node ge(Node node);

            Node lt(Node node);

            Node le(Node node);

            Node eq(Node node);

            Node neq(Node node);

            Node approx_eq(Node node, double tol = 0.000001);

            Node approx_neq(Node node, double tol = 0.000001);

            Node is_nan();

            Node is_inf();

            Node all();

            Node any();

            Node select(Node result_true, Node result_false);

//            Node zero_elem();
//
//            Node non_zero_elem();

            Node exp();

            Node log();

            Node log10();

            Node log1p();

            /**
             * The threshold is used for condition.
             * If x > threshold than we return x, otherwise log1p(e^x)
             * If you want to remove this check set the threshold to 0.
             */
            Node softplus(int threshold = 50);

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

            Node transpose();

            static Node dot(NodeVec nodes);

            static Node dot(Node node1, Node node2);

            Node minv();

            Node det();

            Node logdet();

            Node trace();

            Node diag();

            Node reshape(Shape shape);

            Node flatten(unsigned short dims = 1);

            Node reorder(Axes order);

            Node reorder(short dim0, short dim1, short dim2 = 2, short dim3 = 3);

            Node slice(Node index, short axis = 0);

            Node index(Node index, short axis = AUTO_INFER_AXIS);

            Node max(short axis = AUTO_INFER_AXIS);

            Node argMax(short axis = AUTO_INFER_AXIS);

            std::pair<Node, Node> maxAndArgMax(short axis = AUTO_INFER_AXIS);

            Node sort(short axis = AUTO_INFER_AXIS);

            Node argSort(short axis = AUTO_INFER_AXIS);

            std::pair<Node, Node> sortAndArgSort(short axis = AUTO_INFER_AXIS);

            Node binary_cross_entropy_logit(Node node);

        };

        /** Abstract class for operators */
        class Operator {
        public:
            std::shared_ptr<spdlog::logger> logger() const {
                return metadiff::logger("operator::" + this->name);
            }

            void throw_error(std::exception const & err){
                logger()->error() << name << "] " << err.what();
                throw err;
            }
        public:
            /** Pointer to the owning GraphInternal */
            GraphInPtr graph;
            /** Pointer to the owning Node */
            Node owner;
            /** Unique name of the concrete Operator class */
            std::string const name;

            Operator(std::string name,
                     GraphInPtr graph) :
                    name(name),
                    graph(graph) { };

            /** Copies the operator to a new graph, by using the ancestors
             * provided from the new graph. See Node::copy_to(GraphInPtr graph, NodeVec ancestors)
             * */
            virtual std::shared_ptr<Operator> copy_to(GraphInPtr graph, NodeVec ancestors) const = 0;

            /** Calculates what should be the resulting NodeInternal#dtype */
            virtual dType get_dtype() const = 0;

            /** Calculates what should be the resulting NodeInternal#shape */
            virtual Shape get_shape() const = 0;

            /** Calculates what should be the resulting NodeInternal#node_type */
            virtual nodeType get_node_type() const = 0;

            /** Calculates what should be the resulting NodeInternal#grad_level */
            virtual size_t get_grad_level() const = 0;

            /** Returns the parents NodeVec of this operator */
            virtual NodeVec get_parents() const = 0;

            /** Returns the arguments NodeVec of this operator */
            virtual NodeVec get_arguments() const = 0;

            /**
             * A function which should compute and return the gradient with respect
             * to the parent and the specified index, given the gradient of the owner node
             */
            virtual Node get_parent_grad(Node my_grad, unsigned short index) = 0;

            /**
             * Sends a gradient message from this Operator to the parent with id target.
             * If the target has no gradient messages, then just inserts the new message,
             * otherwise it adds it to the already existing message
             * (e.g. accumulates the gradients)
             *
             * See: generate_gradients(NodeVec &messages)
             */
            void send_grad_message(size_t target,
                                   Node msg,
                                   NodeVec &messages) const;

            /** Generates gradient messages for all parents of this Operator.
             *
             * See: send_grad_message(size_t target, Node msg, NodeVec &messages)
             */
            virtual void generate_gradients(NodeVec &messages);

            /**
             * TODO this and the symbolic_equals are things which aren't yet well done
             * Compares only if this operator is equal to the other, not the other way around.
             * Note that although equality is symmetric, because of mathematical idenitities
             * and the fact that the code is with each operator separately the true equality
             * operator is `op1.equals(op2) or op2.equals(op1)`
             */
            virtual bool equals(std::shared_ptr<const Operator> const op) const = 0;

            /**
             * Returns the union of the parents and arguments of this Operator
             *
             * See: get_parents(), get_ancestors()
             */
            NodeVec get_ancestors() const;

            /**
             * Skips any alias operators to get the base operator
             */
            static std::shared_ptr<Operator> get_base_op(std::shared_ptr<const Operator> const op);
        };

        /**
         * This class stores all of the data for each single node of the GraphInternal
         */
        class NodeInternal {
        public:
            GraphInPtr graph;
            Device device;
            size_t id;
            std::string name;
            nodeType node_type;
            dType dtype;
            Shape shape;
            std::shared_ptr<Operator> op;
            NodeVec children;
            size_t grad_level;
            // Value variables (how can we make this better?)
//        SymInt sym_value;
//        af::array value;
//        SharedPtr shared;
            // Data populated by the optimizer
            ExecutionData execution;
            Group group;

            NodeInternal(GraphInPtr graph, Device device) :
                    graph(graph),
                    device(device) { }

            NodeInternal(GraphInPtr graph,
                         Device device,
                         size_t id,
                         std::string name,
                         nodeType node_type,
                         dType dtype,
                         Shape shape,
                         std::shared_ptr<Operator> op,
                         size_t grad_level,
                         Group group) :
                    graph(graph),
                    device(device),
                    id(id),
                    name(name),
                    node_type(node_type),
                    dtype(dtype),
                    op(op),
                    grad_level(grad_level),
                    shape(shape),
                    group(group) { }
        };

        /**
         * The internal computation graph class
         * TODO: Should think what to be made private
         * TODO: Should add an ordering to the computation, so that it does not
         * necessarily follows the order of creation of the variables.
         */
        class GraphInternal : public std::enable_shared_from_this<GraphInternal> {
        private:
            std::shared_ptr<spdlog::logger> logger() const {
                return metadiff::logger("graph");
            }
        public:
            /** The name of the graph */
            std::string name;
            /** The default device to use for the graph */
            Device default_device;
            /** The maximum floating point precision to allow (See #dType) */
            dType max_float;
            /** The maximum integer precision to allow (See #dType) */
            dType max_int;
            /** Type promotion function. See ::default_dType_promotion(dType type1,
                                      dType type2,
                                      dType max_float,
                                      dType max_int);*/
            std::function<dType(dType type1, dType type2)> promote_type;

            /** Error policy for implicit broadcasts */
            errorPolicy broadcast_err_policy;
            /** Error policy for type promotions */
            errorPolicy type_promotion_err_policy;
            /** Error policy for implicit cast */
            errorPolicy cast_err_policy;


            size_t sym_integer_count;
            std::vector<std::shared_ptr<NodeInternal>> nodes;
            std::vector<SharedPtr> shared_vars;
            Updates updates;

            std::vector<std::shared_ptr<NodeGroup>> groups;
            size_t gradient_mode;
            Group current_group;

            NodeVec temporary_constants;
            Updates temporary_updates;

            GraphInternal() {
                // TODO Have a better preference of devices available in order
                name = "Function";
                sym_integer_count = 0;
                default_device = Device(HOST, 0);
                max_float = f32;
                max_int = i32;
                promote_type = [this](dType dtype1, dType dtype2)->dType {
                    return default_dType_promotion(dtype1, dtype2, this->max_float, this->max_int);
                };
                broadcast_err_policy = WARN;
                type_promotion_err_policy = WARN;
                cast_err_policy = WARN;
                groups.push_back(std::make_shared<NodeGroup>());
                gradient_mode = 0;
                current_group = groups[0];
            }

            /** Checks if the corresponding NodeInternal is in #temporary_constants. */
            bool is_temporary_constant(Node node) const;

            /** Copies the computations with value `true` in the mask to the new_graph */
            NodeVec copy(GraphInPtr new_graph, std::vector<bool> mask) const;

            /** Returns an array masking all descendants of the marked nodes */
            std::vector<bool> get_descendants_mask(NodeVec marked) const;

            /** Returns an array masking all ancestors of the marked nodes */
            std::vector<bool> get_ancestors_mask(NodeVec marked) const;

            /**
             * Finds a node which performs the same operation
             * TODO Not implemented correctly
             */
            Node find_same_node(std::shared_ptr<Operator> op);

            /** Adds the updates to the temporary updates of the graph */
            void add_temporary_updates(Updates const &updates);

            /** Removes all temporary updates of the graph */
            void clear_temporary_updates();

            /** Returns the gradients of the objective with respect to the parameters provided */
            NodeVec gradient(Node objective, NodeVec params);

            /** Optimizes a graph with respect to the given nodes (INTERNAL) */
            Graph optimize(NodeVec &targets, Updates &updates, NodeVec &inputs,
                           NodeVec &new_targets, Updates &new_updates, NodeVec &new_inputs);

            /**
             * Creates a new shared variable
             * TODO This should be made independent from arrayfire as well as the whole SharedVariable class
             */
            Node shared_var(af::array value, std::string name = "SharedVar");

            /** Creates a new derived node (INTERNAL) */
            Node derived_node(std::shared_ptr<Operator> op);

            /** Adds an update for the shared node */
            void update_node(Node shared, Node update);

//            /**
//             * Creates a new constant node
//             * TODO This should be made independant from arrayfire as well as the whole SharedVariable class
//             */
//        Node constant_node(af::array value);
            /** Returns the next unused symbolic integer */
            SymInt get_new_symbolic_integer();

            /** Returns the group specified by full_name. If it does not exist creates it. */
            Group get_group(std::string full_name);

            /** Sets the current group to the specified by the name. If it does not exists creates it */
            void set_group(std::string full_name);

            /**
             * Sets the current group to the group specified by base_name and its parent */
            void set_group(std::string base_name, Group parent);

            /** Sets the current group to #GROUP_ROOT */
            void reset_group();

            /** Returns a Node representing 'pi', with the maximum allowed floating point precision */
            Node PI();

            /** Returns a Node representing 'e', with the maximum allowed floating point precision */
            Node E();

            /** Returns a Node representing ln(2), with the maximum allowed floating point precision */
            Node LN_2();

            /** Returns a Node representing ln(10), with the maximum allowed floating point precision */
            Node LN_10();

            /** Creates a four dimensional #INPUT variable */
            Node tensor4(dType v_type,
                              std::array<SymInt, 4> shape,
                              std::string name = "InputTensor");

            /** Creates a four dimensional #INPUT variable */
            Node tensor4(dType v_type,
                              SymInt shape0,
                              SymInt shape1,
                              SymInt shape2,
                              SymInt shape3,
                         std::string name = "InputTensor");

            /** Creates a four dimensional #INPUT variable */
            Node tensor4(dType v_type,
                         std::string name = "InputTensor");

            /** Creates a four dimensional #INPUT variable, with the same specs as the one provided */
            Node tensor4_as(Node node,
                            std::string name = "InputTensor");

            /** Creates a three dimensional #INPUT variable */
            Node tensor3(dType v_type,
                         std::array<SymInt, 3> shape,
                         std::string name = "InputTensor3");

            /** Creates a three dimensional #INPUT variable */
            Node tensor3(dType v_type,
                              SymInt shape0,
                              SymInt shape1,
                              SymInt shape2,
                         std::string name = "InputTensor3");

            /** Creates a three dimensional #INPUT variable */
            Node tensor3(dType v_type,
                         std::string name = "InputTensor3");

            /** Creates a three dimensional #INPUT variable, with the same specs as the one provided */
            Node tensor3_as(Node node,
                            std::string name = "InputTensor3");

            /** Creates an #INPUT matrix  */
            Node matrix(dType v_type,
                        std::array<SymInt, 2> shape,
                        std::string name = "InputMatrix");

            /** Creates an #INPUT matrix  */
            Node matrix(dType v_type,
                             SymInt shape0,
                             SymInt shape1,
                        std::string name = "InputMatrix");

            /** Creates an #INPUT matrix  */
            Node matrix(dType v_type,
                        std::string name = "InputMatrix");

            /** Creates an #INPUT matrix, with the same specs as the one provided */
            Node matrix_as(Node node,
                           std::string name = "InputMatrix");

            /** Creates a square #INPUT matrix  */
            Node square_matrix(dType v_type,
                                    SymInt shape,
                               std::string name = "InputMatrix");

            /** Creates an #INPUT vector  */
            Node vector(dType v_type,
                             SymInt shape,
                        std::string name = "InputVector");

            /** Creates an #INPUT vector  */
            Node vector(dType dtype,
                        std::string name = "InputVector");

            /** Creates an #INPUT vector, with the same specs as the one provided */
            Node vector_as(Node node,
                           std::string name = "InputVector");

            /** Creates an #INPUT scalar */
            Node scalar(dType dtype,
                        std::string name = "InputScalar");

            /** Returns an identity matrix of the given dimension size */
            Node eye(SymInt size, dType type);

            /** Returns an identity matrix of the given dimension size */
            Node eye(SymInt size);

            /** Returns a matrix filled with ones with the given shape */
            Node ones(Shape shape, dType type);

            /** Returns a matrix filled with ones with the given shape */
            Node ones(Shape shape);

            /** Returns a matrix filled with zeros with the given shape */
            Node zeros(Shape shape, dType type);

            /** Returns a matrix filled with zeros with the given shape */
            Node zeros(Shape shape);

            /**
             * Returns a Node wrapper around the constant value.
             * The #ones(Shape shape, dType type = f32) is a case of this when value = 1.0
             */
            Node constant_value(double value, Shape shape = {1, 1, 1, 1});

            /**
             * Returns a Node wrapper around the constant value.
             * The #ones(Shape shape, dType type) is a case of this when value = 1.0
             */
            Node constant_value(float value, Shape shape = {1, 1, 1, 1});

            /**
             * Returns a Node wrapper around the constant value.
             * The #ones(Shape shape, dType type) is a case of this when value = 1.0
             */
            Node constant_value(long value, Shape shape = {1, 1, 1, 1});

            /**
             * Returns a Node wrapper around the constant value.
             * The #ones(Shape shape, dType type) is a case of this when value = 1.0
             */
            Node constant_value(int value, Shape shape = {1, 1, 1, 1});

            /**
             * Returns a Node wrapper around the constant value.
             * The #ones(Shape shape, dType type) is a case of this when value = 1.0
             */
            Node constant_value(short value, Shape shape = {1, 1, 1, 1});

            /**
             * Returns a Node wrapper around the boolean value.
             */
            Node constant_value(bool value, Shape shape = {1, 1, 1, 1});

            /**
             * Returns a Node wrapper around the af::array.
             */
            Node constant_value(af::array value);

            /**
             * Returns a Node wrapper around the SymInt
             */
            Node wrap_symbolic_int(SymInt value);

            template<typename T>
            Node wrap(T value) {
                if (std::is_same<T, Node>::value) {
                    return static_cast<Node>(value);
                } else if (std::is_same<T, af::array>::value) {
                    return constant_value(value);
                } else if (std::is_same<T, SymInt>::value) {
                    return wrap_symbolic_int(value);
                } else {
                    return constant_value(value);
                }
            }

            /**
             * Returns a vector representing the sequence from start to end.
             */
            Node seq(SymInt start, SymInt end, dType dtype);

            /**
             * Returns a vector representing the sequence from start to end.
             */
            Node seq(SymInt start, SymInt end);
        };

        /**
         * Check if two nodes are symbolically equivalent.
         * TODO still need to think how to this correctly
         */
        bool symbolic_equals(Node const &node1, Node const &node2);


        /** Convenience for applying an unary operator for a derived node */
        template<typename T>
        Node apply(Node node) {
            return node.unwrap()->graph->derived_node(std::make_shared<T>(node.unwrap()->graph, node));
        }

        /** Convenience for applying a binary operator trough template */
        template<typename T>
        Node apply(Node parent1, Node parent2) {
            GraphInPtr graph = parent1.unwrap()->graph;
            return graph->derived_node(std::make_shared<T>(graph, parent1, parent2));
        }

        /** Convenience for applying a nary operator trough template */
        template<typename T>
        Node apply(NodeVec parents) {
            GraphInPtr graph = parents[0].unwrap()->graph;
            return graph->derived_node(std::make_shared<T>(graph, parents));
        }

        std::string to_string(nodeType node_type) {
            switch (node_type) {
//            case ad_node_type::SYMBOLIC_INTEGER: return "SYMBOLIC_INTEGER";
                case CONSTANT:
                    return "Const";
                case INPUT :
                    return "Input";
//            case ad_node_type::SHARED_INPUT : return "SHARED";
                case INPUT_DERIVED:
                    return "Derived";
                case CONSTANT_DERIVED:
                    return "constDerived";
                default:
                    return "UNREACHABLE";
            }
        }

        std::string to_string(dType dType) {
            if (dType == f64) {
                return "f64";
            } else if (dType == f32) {
                return "f32";
            } else if (dType == f16) {
                return "f16";
            } else if (dType == f8) {
                return "f8";
            } else if (dType == i64) {
                return "i64";
            } else if (dType == i32) {
                return "i32";
            } else if (dType == i16) {
                return "i16";
            } else if (dType == i8) {
                return "i8";
            } else if (dType == u64) {
                return "u64";
            } else if (dType == u32) {
                return "u32";
            } else if (dType == u16) {
                return "u16";
            } else if (dType == u8) {
                return "u8";
            } else if (dType == b8) {
                return "bool";
            } else {
                return "UNREACHABLE";
            }
        }

//    std::string to_string(ad_dType const & type){
//        switch(type){
//            case ad_dType::FLOAT: return "FLOAT";
//            case ad_dType::INTEGER: return "INTEGER";
//            case ad_dType::BOOLEAN: return "BOOLEAN";
//        }
//        return "UNREACHABLE";
//    }

        std::string to_string(deviceType type) {
            switch (type) {
                case HOST:
                    return "HOST";
                case GPU:
                    return "GPU";
                default:
                    return "UNREACHABLE";
            }
        }

        std::string to_string(errorPolicy policy) {
            switch (policy) {
                case RAISE:
                    return "Raise";
                case WARN:
                    return "Warn";
                case QUIET:
                    return "Quiet";
                default:
                    return "UNREACHABLE";
            }
        }

//    std::string to_string(ad_float_type const & type){
//        switch(type){
//            case ad_float_type::f16: return "f16";
//            case ad_float_type::f32: return "f32";
//            case ad_float_type::f64: return "f64";
//        }
//        return "UNREACHABLE";
//    }

//    std::string to_string(ad_integer_type const & type){
//        switch(type){
//            case ad_integer_type::i8: return "i8";
//            case ad_integer_type::i16: return "i16";
//            case ad_integer_type::i32: return "i32";
//            case ad_integer_type::i64: return "i64";
//            case ad_integer_type::s8: return "s8";
//            case ad_integer_type::u8: return "u8";
//            case ad_integer_type::s16: return "s16";
//            case ad_integer_type::u16: return "u16";
//            case ad_integer_type::s32: return "s32";
//            case ad_integer_type::u32: return "u32";
//            case ad_integer_type::s64: return "s64";
//            case ad_integer_type::u64: return "u64";
//        }
//        return "UNREACHABLE";
//    }

        std::string to_string(Device const &device) {
            return to_string(device.type) + "[" + std::to_string(device.id) + "]";
        }

        std::ostream &operator<<(std::ostream &f, nodeType node_type) {
            f << to_string(node_type);
            return f;
        }

        std::ostream &operator<<(std::ostream &f, dType dType) {
            f << to_string(dType);
            return f;
        }

        std::ostream &operator<<(std::ostream &f, deviceType deviceType) {
            f << to_string(deviceType);
            return f;
        }

        std::ostream &operator<<(std::ostream &f, errorPolicy policy) {
            f << to_string(policy);
            return f;
        }

//    std::ostream & operator<<(std::ostream & f, ad_float_type const & type) {
//        f << to_string(type);
//        return f;
//    }
//
//    std::ostream & operator<<(std::ostream & f, ad_integer_type const & type) {
//        f << to_string(type);
//        return f;
//    }

        std::ostream &operator<<(std::ostream &f, Device const &device) {
            f << to_string(device);
            return f;
        }
    }

}

#endif //METADIFF_CORE_H