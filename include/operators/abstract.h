//
// Created by alex on 04/05/16.
//

#ifndef METADIFF_ABSTRACT_H
#define METADIFF_ABSTRACT_H
namespace metadiff {
    namespace op {
        using namespace core;
        using namespace exceptions;

        /**
        * Helper function to validate axes argument
        * Checks if each axes is a distinct integer between [0,3]
        */
        bool validate_axes(Axes axes) {
            if (axes.size() > 4) {
                return false;
            }
            bool checks[4]{false, false, false, false};
            for (int i = 0; i < axes.size(); i++) {
                if (axes[i] > 3) {
                    return false;
                }
                if (checks[axes[i]]) {
                    return false;
                }
                checks[axes[i]] = true;
            }
            return true;
        }

        /**
         * Helper function to verify shapes of elementwise operators
         * Verifies that the shapes of the inputs are equal
         */
        Shape verify_elementwise_shapes(std::string name,
                                        NodeVec node_ptrs,
                                        std::shared_ptr<spdlog::logger> const logger) {
            Shape max_shape = node_ptrs[0]->shape;
            for (int i = 1; i < node_ptrs.size(); i++) {
                Shape node_shape = node_ptrs[i]->shape;
                bool max = false;
                for (int j = 0; j < 4; j++) {
                    if (node_shape[j] != 1 and max_shape[j] == 1) {
                        max = true;
                        break;
                    }
                }
                if (max) {
                    for (int j = 0; j < 4; j++) {
                        if (node_shape[j] == 1 and max_shape[j] != 1) {
                            auto err = IncompatibleShapes(node_ptrs, name);
                            logger->error() << err.msg;
                            throw err;
                        } else if (node_shape[j] != 1 and max_shape[j] != 1 and node_shape[j] != max_shape[j]) {
                            auto err = IncompatibleShapes(node_ptrs, name);
                            logger->error() << err.msg;
                            throw err;
                        }
                    }
                    max_shape = node_shape;
                }
            }
            return max_shape;
        }

        /** Abstract class for unary operators */
        class UnaryOperator : public Operator {
        public:
            Node parent;

            UnaryOperator(std::string const name,
                          GraphInPtr graph,
                          Node parent) :
                    Operator(name, graph),
                    parent(parent) { };

            NodeVec get_parents() const {
                return {parent};
            };

            void replace_parent(Node iOrg, Node iNew) {
                parent = iNew;
            }

            dType get_dtype() const {
                return parent->dtype;
            };

            nodeType get_node_type() const {
                if (parent->node_type == INPUT
                    or parent->node_type == INPUT_DERIVED) {
                    return INPUT_DERIVED;
                } else {
                    return CONSTANT_DERIVED;
                }
            };

            Shape get_shape() const {
                return parent->shape;
            }

            unsigned short get_grad_level() const {
                return parent->grad_level;
            };

            NodeVec get_arguments() const {
                return NodeVec {};
            }

            bool equals(std::shared_ptr<const Operator> const op) const {
                if (name == op->name) {
                    auto cast_op = std::static_pointer_cast<const UnaryOperator>(op);
                    return symbolic_equals(parent, cast_op->parent);
                }
                return false;
            }
        };

        /** Abstract class for binary operators. */
        class BinaryOperator : public Operator {
        public:
            Node parent1;
            Node parent2;
            Shape shape;

            BinaryOperator(std::string const name,
                           GraphInPtr graph,
                           Node parent1,
                           Node parent2) :
                    Operator(name, graph),
                    parent1(parent1),
                    parent2(parent2) { }

            NodeVec get_parents() const {
                return {parent1, parent2};
            };

            void replace_parent(Node iOrg, Node iNew) {
                if (parent1->id == iOrg->id) parent1 = iNew;
                if (parent2->id == iOrg->id) parent2 = iNew;
            }

            dType get_dtype() const {
                return graph->promote_type(parent1->dtype, parent2->dtype);
            };

            nodeType get_node_type() const {
                if (parent1->node_type == INPUT
                    or parent1->node_type == INPUT_DERIVED
                    or parent2->node_type == INPUT
                    or parent2->node_type == INPUT_DERIVED) {
                    return INPUT_DERIVED;
                } else {
                    return CONSTANT_DERIVED;
                }
            };

            Shape get_shape() const {
                return shape;
            }

            unsigned short get_grad_level() const {
                return parent1->grad_level > parent2->grad_level ?
                       parent1->grad_level :
                       parent2->grad_level;
            };

            NodeVec get_arguments() const {
                return NodeVec {};
            }

            bool equals(std::shared_ptr<const Operator> const op) const {
                if (name == op->name) {
                    auto cast_op = std::static_pointer_cast<const BinaryOperator>(op);
                    return symbolic_equals(parent1, cast_op->parent1) and
                           symbolic_equals(parent2, cast_op->parent2);
                }
                return false;
            }

        };

        /** Abstract class for any operators that take 2 or more arguments */
        class NaryOperator : public Operator {
        public:
            NodeVec parents;
            Shape shape;

            NaryOperator(std::string const name,
                         GraphInPtr graph,
                         NodeVec parents) :
                    Operator(name, graph),
                    parents(parents) {
                if (parents.size() < 2) {
                    auto err = InvalidArguments(parents, name, "All NaryOperators require at least 2 parents");
                    logger()->error() << err.msg;
                    throw err;
                }
            };

            NodeVec get_parents() const {
                return parents;
            };

            void replace_parent(Node iOrg, Node iNew) {
                for (auto& p : parents) 
                {
                    if (p->id == iOrg->id) p = iNew;
                }
            }

            void update_parents(NodeVec nodes) {
                parents = nodes;
            }

            dType get_dtype() const {
                dType dtype = b8;
                for (size_t i = 0; i < parents.size(); i++) {
                    dtype = graph->promote_type(dtype, parents[i]->dtype);
                }
                return dtype;
            };

            nodeType get_node_type() const {
                for (int i = 0; i < parents.size(); i++) {
                    if (parents[i]->node_type == INPUT
                        or parents[i]->node_type == INPUT_DERIVED) {
                        return INPUT_DERIVED;
                    }
                }
                return CONSTANT_DERIVED;
            };

            Shape get_shape() const {
                return shape;
            }

            unsigned short get_grad_level() const {
                size_t max_grad_level = 0;
                for (int i = 0; i < parents.size(); i++) {
                    if (parents[i]->grad_level > max_grad_level) {
                        max_grad_level = parents[i]->grad_level;
                    }
                }
                return max_grad_level;
            };

            NodeVec get_arguments() const {
                return NodeVec {};
            }
        };

        /** Abstract class for operators which are constant expressions */
        class ConstantOperator : public Operator {
        public:
            Shape shape;
            dType dtype;

            ConstantOperator(std::string const name,
                             GraphInPtr graph,
                             dType dtype) :
                    Operator(name, graph),
                    dtype(dtype) { };

            ConstantOperator(std::string const name,
                             GraphInPtr graph,
                             Shape shape,
                             dType dtype) :
                    Operator(name, graph),
                    shape(shape),
                    dtype(dtype) { };

            NodeVec get_parents() const {
                return {};
            };

            dType get_dtype() const {
                return dtype;
            };

            nodeType get_node_type() const {
                for (int i = 0; i < 4; i++) {
                    if (not shape[i].is_constant()) {
                        return CONSTANT_DERIVED;
                    }
                }
                return CONSTANT;
            };

            Shape get_shape() const {
                return shape;
            }

            unsigned short get_grad_level() const {
                return 0;
            };

            NodeVec get_arguments() const {
                return NodeVec {};
            }

            Node get_parent_grad(Node my_grad, unsigned short index) {
                auto err = WrongGradient(NodeVec{owner, my_grad}, name);
                logger()->error() << err.msg;
                throw err;
            }

            bool equals(std::shared_ptr<const Operator> const op) const {
                if (name == op->name) {
                    auto cast_op = std::static_pointer_cast<const ConstantOperator>(op);
                    return shape == cast_op->shape and dtype == cast_op->dtype;
                }
                return false;
            }
        };

        /** Abstract class for binary operators which are applied elementwise */
        class ElementwiseBinary : public BinaryOperator {
        public:
            ElementwiseBinary(std::string const name,
                              GraphInPtr graph,
                              Node parent1,
                              Node parent2) :
                    BinaryOperator(name, graph, parent1, parent2) {
                NodeVec parents = get_parents();
                shape = verify_elementwise_shapes(name, NodeVec{parents}, logger());
                if (parent1->shape != shape and not parent1.is_scalar()) {
                    operate_policy(graph->broadcast_err_policy, logger(),
                                         ImplicitBroadcast(NodeVec{parent1, parent2}, name));
                    this->parent1 = parent1.broadcast(shape);
                }
                if (parent2->shape != shape and not parent2.is_scalar()) {
                    operate_policy(graph->broadcast_err_policy, logger(),
                                         ImplicitBroadcast(NodeVec{parent1, parent2}, name));
                    this->parent2 = parent2.broadcast(shape);
                }
            }
        };

        /** Abstract class for nary operators which are applied elementwise */
        class ElementwiseNary : public NaryOperator {
        public:
            ElementwiseNary(std::string const name,
                            GraphInPtr graph,
                            NodeVec parents) :
                    NaryOperator(name, graph, parents) {
                this->parents.clear();
                shape = verify_elementwise_shapes(name, parents, logger());
                for (int i = 0; i < parents.size(); i++) {
                    if (parents[i]->shape != shape and not parents[i].is_scalar()) {
                        operate_policy(graph->broadcast_err_policy, logger(),
                                             ImplicitBroadcast(parents, name));
                        this->parents.push_back(parents[i].broadcast(shape));
                    } else {
                        this->parents.push_back(parents[i]);
                    }
                }
            };
        };

        /** Abstract class for unary logical operators */
        class LogicalUnary : public UnaryOperator {
        public:
            LogicalUnary(std::string const name,
                         GraphInPtr graph,
                         Node parent) :
                    UnaryOperator(name, graph, parent) {};

            dType get_dtype() const {
                return b8;
            };

            nodeType get_node_type() const {
                return CONSTANT_DERIVED;
            };

            Node get_parent_grad(Node my_grad, unsigned short index) {
                auto err = WrongGradient(NodeVec{owner, my_grad}, name);
                logger()->error() << err.msg;
                throw err;
            }
        };

        /** Abstract class for binary logical operators */
        class LogicalBinary : public ElementwiseBinary {
        public:
            LogicalBinary(std::string const name,
                          GraphInPtr graph,
                          Node parent1,
                          Node parent2) :
                    ElementwiseBinary(name, graph, parent1, parent2) { };

            dType get_dtype() const {
                return b8;
            };

            nodeType get_node_type() const {
                return CONSTANT_DERIVED;
            };

            Node get_parent_grad(Node my_grad, unsigned short index) {
                auto err = WrongGradient(NodeVec{owner, my_grad}, name);
                logger()->error() << err.msg;
                throw err;
            }
        };
    }
}

#endif //METADIFF_ABSTRACT_H
