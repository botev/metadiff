//
// Created by alex on 10/12/15.
//

#ifndef METADIFF_CORE_H
#define METADIFF_CORE_H

namespace metadiff {
//    template <typename R, typename = std::enable_if<not std::is_same<R, Node>::value>>
    namespace core {
        using shared::SharedPtr;

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

        /** Helper function for calculating the number of elements of a tensor */
        SymInt number_of_elements(Shape shape){
            return (shape[0] * shape[1]) * (shape[2] * shape[3]);
        }

        /** The class is an API wrapper around a NodeInternal */
        class Node {
        private:
            std::shared_ptr<spdlog::logger> logger() const;
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

            Node neg();

            static Node mul(NodeVec nodes);

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

            Node T(){
                return transpose();
            }

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

//            Node slice(Node index, short axis = 0);
//
//            Node index(Node index, short axis = AUTO_INFER_AXIS);

            Node max(short axis = AUTO_INFER_AXIS);

            Node argMax(short axis = AUTO_INFER_AXIS);

            std::pair<Node, Node> maxAndArgMax(short axis = AUTO_INFER_AXIS);

            Node sort(short axis = AUTO_INFER_AXIS);

            Node argSort(short axis = AUTO_INFER_AXIS);

            std::pair<Node, Node> sortAndArgSort(short axis = AUTO_INFER_AXIS);

            Node binary_cross_entropy_logit(Node node);

            //below are graph optimization utilities
            void remove_child(Node node);

            void replace_children_from(Node node);

            void replace_parent_of_children(Node node);

            Node replace_with_constant(double value);

            void replace_const_eli(int value, Node parent);

            bool is(const string& str);

            void set_inactive();

            bool is_active();
        };

        /** Abstract class for operators */
        class Operator {
        public:
            std::shared_ptr<spdlog::logger> logger() const {
                return logging::logger("operator::" + this->name);
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
            virtual unsigned short get_grad_level() const = 0;

            /** Returns the parents NodeVec of this operator */
            virtual NodeVec get_parents() const = 0;

            /** get parents id in ids */
            void get_parents_ids(std::vector<int>& ids) const;

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
            static std::shared_ptr<const Operator> get_base_op(std::shared_ptr<const Operator> const op);

            //below are graph optimization utilities

            // only for NaryOperator
            virtual void replace_parent(Node iOrg, Node iNew) {
              logger()->error()<<"replace_parent for NaryOperator is called for other type!";
            };

            // only for NaryOperator
            virtual void update_parents(NodeVec nodes) {
              logger()->error()<<"update_parents for NaryOperator is called for other type!";
            };

            // override for ConstantValue class
            virtual double getConstVal() {return 0.0;};
        };

        /**
         * This class stores all of the data for each single node of the GraphInternal
         */
        class NodeInternal {
        public:
            GraphInPtr graph;
            size_t id;
            std::string name;
            Group group;
            Device device;
            nodeType node_type;
            dType dtype;
            Shape shape;
            std::shared_ptr<Operator> op;
            NodeVec children;
            unsigned short grad_level;
            // Data populated by the optimizer
            ExecutionData execution;
            // graph optimization variable - to remove inactive nodes
            bool active;

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
                         unsigned short grad_level,
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
                    group(group),
                    active(true) { }
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
                return logging::logger("graph::" + name);
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
            Updates updates;

            std::vector<std::shared_ptr<NodeGroup>> groups;
            unsigned short grad_level;
            Group current_group;

            NodeVec temporary_constants;
            Updates temporary_updates;

            GraphInternal() {
                // TODO Have a better preference of devices available in order
                name = "Function";
                sym_integer_count = 0;
                default_device = MASTER;
                max_float = f32;
                max_int = i32;
                promote_type = [this](dType dtype1, dType dtype2)->dType {
                    return default_dType_promotion(dtype1, dtype2, this->max_float, this->max_int);
                };
                broadcast_err_policy = WARN;
                type_promotion_err_policy = WARN;
                cast_err_policy = WARN;
                groups.push_back(std::make_shared<NodeGroup>());
                grad_level = 0;
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
            Graph optimize(const NodeVec &targets,
              const Updates &updates,
              const NodeVec &inputs,
              NodeVec &new_targets, 
              Updates &new_updates,
              NodeVec &new_inputs);

            /** Creates a new derived node (INTERNAL) */
            Node derived_node(std::shared_ptr<Operator> op);

            /** Adds an update for the shared node */
            void update_node(Node shared, Node update);

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

            /** Returns a Node wrapper around a shared variable */
            Node shared_variable(SharedPtr var, std::string name = "SharedVar");
#ifdef AFAPI
            /** Returns a Node wrapper around a shared variable */
            Node shared_variable(af::array value, std::string name = "SharedVar");
#endif
            /** Returns a Node wrapper around the value. */
            Node constant_value(bool value, Shape shape = scalar_shape);

            /** Returns a Node wrapper around the short value. */
            Node constant_value(unsigned short value, Shape shape = scalar_shape);

            /** Returns a Node wrapper around the int value. */
            Node constant_value(unsigned int value, Shape shape = scalar_shape);

            /** Returns a Node wrapper around the long value. */
            Node constant_value(unsigned long value, Shape shape = scalar_shape);

            /** Returns a Node wrapper around the short value. */
            Node constant_value(short value, Shape shape = scalar_shape);

            /** Returns a Node wrapper around the int value. */
            Node constant_value(int value, Shape shape = scalar_shape);

            /** Returns a Node wrapper around the long value. */
            Node constant_value(long value, Shape shape = scalar_shape);

            /** Returns a Node wrapper around the float value. */
            Node constant_value(float value, Shape shape = scalar_shape);

            /** Returns a Node wrapper around the double value. */
            Node constant_value(double value, Shape shape = scalar_shape);

            Node wrap(Node value){
                return value;
            }

            Node wrap(SharedPtr value){
                return shared_variable(value);
            }

            Node wrap(SymInt value);

            Node wrap(bool value){
                return constant_value(value);
            }

            Node wrap(unsigned short value){
                return constant_value(value);
            }

            Node wrap(unsigned int value){
                return constant_value(value);
            }

            Node wrap(unsigned long value){
                return constant_value(value);
            }

            Node wrap(short value){
                return constant_value(value);
            }

            Node wrap(int value){
                return constant_value(value);
            }

            Node wrap(long value){
                return constant_value(value);
            }

            Node wrap(float value){
                return constant_value(value);
            }

            Node wrap(double value){
                return constant_value(value);
            }
#ifdef AFAPI
            /** Returns a Node wrapper around the af::array. */
            Node constant_value(af::array value);
#endif
            /** Returns a vector representing the sequence from start to end. */
            Node seq(SymInt start, SymInt end, dType dtype);

            /** Returns a vector representing the sequence from start to end. */
            Node seq(SymInt start, SymInt end);

            //below are graph optimization utilities
            void optimize();

            void removeInactiveNodes();

            void removeNode(Node node);
        };

        /**
         * Check if two nodes are symbolically equivalent.
         * TODO still need to think how to this correctly
         */
        bool symbolic_equals(Node const & node1, Node const & node2);


        /** Convenience for applying an unary operator for a derived node */
        template<typename T>
        Node apply(Node node) {
            return node->graph->derived_node(std::make_shared<T>(node->graph, node));
        }

        /** Convenience for applying a binary operator trough template */
        template<typename T>
        Node apply(Node parent1, Node parent2) {
            GraphInPtr graph = parent1->graph;
            return graph->derived_node(std::make_shared<T>(graph, parent1, parent2));
        }

        /** Convenience for applying a nary operator trough template */
        template<typename T>
        Node apply(NodeVec parents) {
            GraphInPtr graph = parents[0]->graph;
            return graph->derived_node(std::make_shared<T>(graph, parents));
        }
    }
}

#endif //METADIFF_CORE_H