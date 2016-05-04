//
// Created by alex on 03/05/16.
//

#ifndef METADIFF_SPECIAL_H
#define METADIFF_SPECIAL_H

namespace metadiff {
    namespace op {
        using namespace core;
        using namespace exceptions;

        /** Operator for input variables */
        class Input : public Operator {
        public:
            dType dtype;

            Input(GraphInPtr graph, dType dtype) :
                    Operator("Input", graph),
                    dtype(dtype) { }

            std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const {
                return std::make_shared<Input>(graph, dtype);
            }

            dType get_dtype() const {
                return dtype;
            }

            Shape get_shape() const {
                return Shape{0, 0, 0, 0};
            }

            nodeType get_node_type() const {
                return INPUT;
            };

            size_t get_grad_level() const {
                return 0;
            }

            NodeVec get_parents() const {
                return NodeVec {};
            }

            NodeVec get_arguments() const {
                return NodeVec {};
            }

            Node get_parent_grad(Node my_grad, unsigned short index) {
                throw WrongGradient(name, {}, {});
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                return false;
            }
        };

        /** Operator for shared input variables */
        class SharedInput : public Operator {
        public:
            SharedPtr value;

            SharedInput(GraphInPtr graph, SharedPtr value) :
                    Operator("Shared", graph),
                    value(value) { }

            std::shared_ptr<Operator> copy_to(GraphInPtr graph, NodeVec ancestors) const {
                return std::make_shared<SharedInput>(graph, value);
            }

            dType get_dtype() const {
                return convert_af_dtype(value->value.type());
            }

            Shape get_shape() const {
                af::dim4 dims = value->value.dims();
                return Shape {dims[0], dims[1], dims[2], dims[3]};
            }

            nodeType get_node_type() const {
                return INPUT;
            };

            size_t get_grad_level() const {
                return 0;
            }

            NodeVec get_parents() const {
                return NodeVec {};
            }

            NodeVec get_arguments() const {
                return NodeVec {};
            }

            Node get_parent_grad(Node my_grad, unsigned short index) {
                throw WrongGradient(name, {}, {});
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                return false;
            }
        };

        /** Operator for wrapping Symbolic Integers */
        class SymIntWrapper : public Operator {
        public:
            SymInt value;

            SymIntWrapper(GraphInPtr graph, SymInt value) :
                    Operator("SymInt", graph),
                    value(value) { }

            std::shared_ptr<Operator> copy_to(GraphInPtr graph, NodeVec ancestors) const {
                return std::make_shared<SymIntWrapper>(graph, value);
            }

            dType get_dtype() const {
                return graph->max_int;
            }

            Shape get_shape() const {
                return Shape{1, 1, 1, 1};
            }

            nodeType get_node_type() const {
                return value.is_constant() ? CONSTANT : CONSTANT_DERIVED;
            };

            size_t get_grad_level() const {
                return 0;
            }

            NodeVec get_parents() const {
                return NodeVec {};
            }

            NodeVec get_arguments() const {
                return NodeVec {};
            }

            Node get_parent_grad(Node my_grad, unsigned short index) {
                throw WrongGradient(name, {}, {});
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                if (op->name == "SymInt") {
                    std::shared_ptr<SymIntWrapper> cast_op = std::static_pointer_cast<SymIntWrapper>(op);
                    return cast_op->value == value;
                }
                return false;
            }
        };
    }

    namespace core {
        Node GraphInternal::wrap_symbolic_int(SymInt value) {
            std::shared_ptr<Operator> op = std::make_shared<op::SymIntWrapper>(this, value);
            return derived_node(op);
        }
    }
}
#endif //METADIFF_SPECIAL_H
