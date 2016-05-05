//
// Created by alex on 13/12/15.
//

#ifndef METADIFF_OPERATORS_BASE_H
#define METADIFF_OPERATORS_BASE_H

namespace metadiff {
    namespace op{
        class Cast : public UnaryOperator{
        public:
            core::dType dtype;
            Cast(core::GraphInPtr graph, core::Node parent, core::dType dtype) :
            UnaryOperator("Cast", graph, parent),
            dtype(dtype) {};

            core::dType get_dtype() const {
                return dtype;
            };

            std::shared_ptr<Operator> copy_to(core::GraphInPtr graph, core::NodeVec ancestors) const {
                return std::make_shared<Cast>(graph, ancestors[0], dtype);
            }

            core::Node get_parent_grad(core::Node my_grad, size_t index) {
                return my_grad.cast(parent->dtype);
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                if (name == op->name) {
                    std::shared_ptr<Cast> cast_op = std::static_pointer_cast<Cast>(op);
                    return symbolic_equals(parent, cast_op->parent) and dtype == cast_op->dtype;
                }
                return false;
            }
        };

        /**
         * Represents an alias for another node
         * This could be particularly useful for multiple device case
         * where an Alias with another device would mean a transfer
         */
        class Alias : public UnaryOperator {
        public:
            Alias(core::GraphInPtr graph, core::Node parent) :
                    UnaryOperator("Alias", graph, parent) { };

            std::shared_ptr<Operator> copy_to(core::GraphInPtr graph, core::NodeVec ancestors) const {
                return std::make_shared<Alias>(graph, ancestors[0]);
            }

            core::Node get_parent_grad(core::Node my_grad, size_t index) {
                return my_grad;
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                std::shared_ptr<Operator> my_op = get_base_op(parent->op);
                return my_op->equals(op) or op->equals(my_op);
            }
        };

        /** Broadcasts the parent to the specified shape */
        class Broadcast : public UnaryOperator {
        public:
            core::Shape to_shape;

            Broadcast(core::GraphInPtr graph,
                      core::Node parent,
                      core::Shape to_shape) :
                    UnaryOperator("Broadcast", graph, parent),
                    to_shape(to_shape) {
                for (int i = 0; i < 4; i++) {
                    if (parent->shape[i] != 1 and parent->shape[i] != to_shape[i]) {
                        auto err = IncompatibleShapes(NodeVec{parent}, name);
                        logger()->error() << name << "] " << err.msg;
                        throw err;
                    }
                }
            }

            std::shared_ptr<Operator> copy_to(core::GraphInPtr graph, core::NodeVec ancestors) const {
                return std::make_shared<Broadcast>(graph, ancestors[0], to_shape);
            }

            core::Shape get_shape() const {
                return to_shape;
            }

            core::Axes get_broadcast_axes() const {
                core::Axes axes;
                auto p1_shape = this->parent->shape;
                for (size_t i = 0; i < 4; i++) {
                    if (p1_shape[i] != to_shape[i]) {
                        axes.push_back(i);
                    }
                }
                return axes;
            }

            core::Node get_parent_grad(core::Node my_grad, size_t index) {
                return my_grad.sum(get_broadcast_axes());
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                if (name == op->name) {
                    std::shared_ptr<Broadcast> cast_op = std::static_pointer_cast<Broadcast>(op);
                    return symbolic_equals(parent, cast_op->parent) and to_shape == cast_op->to_shape;
                }
                return false;
            }
        };


        /** Performs a summation reduction along the axes specified */
        class Sum : public UnaryOperator {
        public:
            core::Axes axes;

            Sum(core::GraphInPtr graph,
                core::Node parent,
                core::Axes axes) :
                    UnaryOperator("Sum", graph, parent),
                    axes(axes) {
                if (not validate_axes(axes)) {
                    std::string axes_str;
                    for (int i = 0; i < axes.size(); i++) {
                        axes_str += std::to_string(axes[i]);
                        if (i < axes.size() - 1) {
                            axes_str += ", ";
                        }
                    }
                    if (axes.size() == 0) {
                        axes_str = "NULL";
                    }
                    auto err = InvalidArguments(NodeVec{parent}, name, "Invalid axes: " + axes_str);
                    logger()->error() << name << "] " << err.msg;
                    throw err;
                }
            }

            std::shared_ptr<Operator> copy_to(core::GraphInPtr graph, core::NodeVec ancestors) const {
                return std::make_shared<Sum>(graph, ancestors[0], axes);
            }

            core::Shape get_shape() const {
                core::Shape p_shape = parent->shape;
                for (int i = 0; i < axes.size(); i++) {
                    p_shape[axes[i]] = 1;
                }
                return p_shape;
            }

            core::Node get_parent_grad(core::Node my_grad, size_t index) {
                return my_grad.broadcast(parent->shape);
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                if (name == op->name) {
                    std::shared_ptr<Sum> cast_op = std::static_pointer_cast<Sum>(op);
                    return symbolic_equals(parent, cast_op->parent) and axes == cast_op->axes;
                }
                return false;
            }

        };

        /** Addition operator */
        class Add : public ElementwiseNary {
        public:
            Add(core::GraphInPtr graph, core::NodeVec parents) :
                    ElementwiseNary("Add", graph, parents) { }

            Add(core::GraphInPtr graph, core::Node parent1, core::Node parent2) :
                    Add(graph, {parent1, parent2}) { }

            std::shared_ptr<Operator> copy_to(core::GraphInPtr graph, core::NodeVec ancestors) const {
                return std::make_shared<Add>(graph, ancestors);
            }

            core::Node get_parent_grad(core::Node my_grad, size_t index) {
                return my_grad;
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                if (name == op->name) {
                    bool check[parents.size()];
                    for (int i = 0; i < parents.size(); i++) {
                        check[i] = false;
                    }
                    if (parents.size() != op->get_parents().size()) {
                        return false;
                    }
                    for (int i = 0; i < parents.size(); i++) {
                        core::Node parent = op->get_parents()[i];
                        int j = 0;
                        for (; j < parents.size(); j++) {
                            if (symbolic_equals(parent, parents[j]) and not check[j]) {
                                check[j] = true;
                                break;
                            }
                        }
                        if (j == parents.size()) {
                            return false;
                        }
                    }
                }
                return false;
            }
        };

        /** Unary negation */
        class Neg : public UnaryOperator {
        public:
            Neg(core::GraphInPtr graph, core::Node parent) :
                    UnaryOperator("Neg", graph, parent) { };

            std::shared_ptr<Operator> copy_to(core::GraphInPtr graph, core::NodeVec ancestors) const {
                return std::make_shared<Neg>(graph, ancestors[0]);
            }

            core::Node get_parent_grad(core::Node my_grad, size_t index) {
                return my_grad.neg();
            };
        };

        /** Elementwise multiplication */
        class Mul : public ElementwiseNary {
        public:
            Mul(core::GraphInPtr graph, core::NodeVec parents) :
                    ElementwiseNary("Mul", graph, parents) { };

            Mul(core::GraphInPtr graph, core::Node p1, core::Node p2) :
                    ElementwiseNary("Mul", graph, {p1, p2}) { };

            std::shared_ptr<Operator> copy_to(core::GraphInPtr graph, core::NodeVec ancestors) const {
                return std::make_shared<Mul>(graph, ancestors);
            }

            core::Node get_parent_grad(core::Node my_grad, size_t index) {
                // TODO change the ones and zeros to correct
                if (parents.size() == 2) {
                    // Special case when only two parents
                    if (my_grad->op->name == "Ones") {
                        return parents[1 - index];
                    } else if (my_grad->op->name == "Zeros") {
                        return my_grad;
                    }
                    if (parents[1 - index]->op->name == "Ones") {
                        return my_grad;
                    } else if (parents[1 - index]->op->name == "Zeros") {
                        return parents[1 - index];
                    }
                    return core::apply<Mul>(my_grad, parents[1 - index]);
                } else {
                    core::Node product = core::apply<Mul>(my_grad, owner);
                    return core::apply<Mul>(product, parents[index].div());
                }
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                if (name == op->name) {
                    bool check[parents.size()];
                    for (int i = 0; i < parents.size(); i++) {
                        check[i] = false;
                    }
                    if (parents.size() != op->get_parents().size()) {
                        return false;
                    }
                    for (int i = 0; i < parents.size(); i++) {
                        core::Node parent = op->get_parents()[i];
                        int j = 0;
                        for (; j < parents.size(); j++) {
                            if (symbolic_equals(parent, parents[j]) and not check[j]) {
                                check[j] = true;
                                break;
                            }
                        }
                        if (j == parents.size()) {
                            return false;
                        }
                    }
                }
                return false;
            }
        };

        /** Unary division (inverse) */
        class Div : public UnaryOperator {
        public:
            Div(core::GraphInPtr graph, core::Node parent) :
                    UnaryOperator("Div", graph, parent) { };

            std::shared_ptr<Operator> copy_to(core::GraphInPtr graph, core::NodeVec ancestors) const {
                return std::make_shared<Div>(graph, ancestors[0]);
            }

            core::Node get_parent_grad(core::Node my_grad, size_t index) {
                return core::Node::mul(core::NodeVec{my_grad, parent.square().div()}).neg();
            }
        };
    }

    namespace core {
        Node Node::cast(dType dtype) {
            GraphInPtr graph = unwrap()->graph;
            return graph->derived_node(std::make_shared<op::Cast>(graph, this, dtype));
        }

        Node Node::alias() {
            return apply<op::Alias>(unwrap());
        }

        Node Node::broadcast(Shape shape) {
            GraphInPtr graph = unwrap()->graph;
            return graph->derived_node(std::make_shared<op::Broadcast>(graph, this, shape));
        }

        Node Node::broadcast_to(Node other) {
            return broadcast(other->shape);
        }

        Node Node::sum(Axes axes) {
            GraphInPtr graph = unwrap()->graph;
            return graph->derived_node(std::make_shared<op::Sum>(graph, this, axes));
        }

        Node Node::add(NodeVec nodes) {
            // TODO a + (-a) = 0
            // TODO a * b + c * b = (a + c) * b ???
            std::vector<size_t> neg_indexes;
            for (size_t i = 0; i < nodes.size(); i++) {
                if (nodes[i]->op->name == "Neg") {
                    neg_indexes.push_back(i);
                }
            }
            if (neg_indexes.size() == 0 or neg_indexes.size() == nodes.size()) {
                return apply<op::Add>(nodes);
            } else {
                NodeVec reordered;
                for (size_t i = 0; i < nodes.size(); i++) {
                    if (std::find(neg_indexes.begin(), neg_indexes.end(), i) == neg_indexes.end()) {
                        reordered.push_back(nodes[i]);
                    }
                }
                for (size_t i = 0; i < neg_indexes.size(); i++) {
                    reordered.push_back(nodes[neg_indexes[i]]);
                }
                return apply<op::Add>(reordered);
            }
        };

        Node Node::add(Node node1, Node node2) {
            return Node::add(NodeVec {node1, node2});
        };

        Node operator+(Node node1, Node node2) {
            return Node::add(NodeVec {node1, node2});
        };

        Node Node::neg() {
            // TODO x.neg().neg() = x
            return apply<op::Neg>(unwrap());
        }

        Node operator-(Node node) {
            return node.neg();
        }

        Node operator-(Node node1, Node node2) {
            return Node::add(NodeVec{node1, node2.neg()});
        }

        Node Node::mul(NodeVec nodes) {
            // TODO e^x * e^y = e^(x+y)
            // TODO x * x = x.square()
            // TODO x * (y / x) = y
            // Reorder so that Div operators are always at the end
            std::vector<size_t> div_indexes;
            for (size_t i = 0; i < nodes.size(); i++) {
                if (nodes[i]->op->name == "Div") {
                    div_indexes.push_back(i);
                }
            }
            if (div_indexes.size() == 0 or div_indexes.size() == nodes.size()) {
                return apply<op::Mul>(nodes);
            } else {
                NodeVec reordered;
                for (size_t i = 0; i < nodes.size(); i++) {
                    if (std::find(div_indexes.begin(), div_indexes.end(), i) == div_indexes.end()) {
                        reordered.push_back(nodes[i]);
                    }
                }
                for (size_t i = 0; i < div_indexes.size(); i++) {
                    reordered.push_back(nodes[div_indexes[i]]);
                }
                return apply<op::Mul>(reordered);
            }

        };

        Node Node::mul(Node node1, Node node2){
            return Node::mul(NodeVec{node1, node2});
        }

        Node operator*(Node node1, Node node2) {
            return Node::mul(NodeVec{node1, node2});
        };

        Node Node::div() {
            // TODO x.div().div() = x
            return apply<op::Div>(unwrap());
        }

        Node operator/(Node node1, Node node2) {
            return Node::mul(NodeVec{node1, node2.div()});
        };
    }
}

#endif //METADIFF_OPERATORS_BASE_H
