//
// Created by alex on 17/12/15.
//

#ifndef METADIFF_OPERATORS_CONSTANTS_H
#define METADIFF_OPERATORS_CONSTANTS_H

namespace metadiff{
    namespace op {
        using namespace core;
        using namespace exceptions;
        
        /** Operator for constant input variables */
        class ConstantInput : public ConstantOperator {
        public:
            af::array value;

            ConstantInput(GraphInPtr graph,
                          af::array value) :
                    ConstantOperator("ConstInput", graph ,
                                     Shape{value.dims(0), value.dims(1),
                                                 value.dims(2), value.dims(3)},
                                     convert_af_dtype(value.type())) {};

            std::shared_ptr<Operator> copy_to(GraphInPtr graph, NodeVec ancestors) const {
                return std::make_shared<ConstantInput>(graph, value);
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                return false;
            }
        };


        /** Tensor filled with the same value */
        class ConstantValue : public ConstantOperator {
        public:
            double value;

            ConstantValue(GraphInPtr graph, double value, Shape shape, dType dtype) :
                    ConstantOperator("Value", graph, shape, dtype),
                    value(value) { };


            std::shared_ptr<Operator> copy_to(GraphInPtr graph, NodeVec ancestors) const {
                return std::make_shared<ConstantValue>(graph, value, shape, dtype);
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                if (ConstantOperator::equals(op)) {
                    std::shared_ptr<ConstantValue> cast_op = std::static_pointer_cast<ConstantValue>(op);
                    return value == cast_op->value;
                } else {
                    return false;
                }
            }
        };

        /** Matrix identity */
        class Eye : public ConstantOperator {
        public:
            Eye(GraphInPtr graph, SymInt size, dType dtype) :
                    ConstantOperator("Eye", graph, Shape{size, size, 1, 1}, dtype) {}

            std::shared_ptr<Operator> copy_to(GraphInPtr graph, NodeVec ancestors) const {
                return std::make_shared<Eye>(graph, shape[0], dtype);
            }
        };


        /** A vector of the sequence from 'start' to 'end' */
        class Sequence : public ConstantOperator {
        public:
            SymInt start;
            SymInt end;

            Sequence(GraphInPtr graph, SymInt start, SymInt end, dType dtype) :
                    ConstantOperator("Sequence", graph, Shape {end - start, 1, 1, 1}, dtype),
                    start(start), end(end) {}

            std::shared_ptr<Operator> copy_to(GraphInPtr graph, NodeVec ancestors) const {
                return std::make_shared<Sequence>(graph, start, end, dtype);
            }

            bool equals(const std::shared_ptr<Operator> op) const {
                if (ConstantOperator::equals(op)) {
                    std::shared_ptr<Sequence> cast_op = std::static_pointer_cast<Sequence>(op);
                    return start == cast_op->start and end == cast_op->end;
                } else {
                    return false;
                }
            }
        };

        /** The operator provides a view of the parent which is constant.
         * This implies that the gradient with respect to the result is always 0. */
        class MakeConstant : public UnaryOperator {
        public:
            MakeConstant(GraphInPtr graph,
                         Node parent) :
                    UnaryOperator("MakeConst", graph, parent) { };

            std::shared_ptr<Operator> copy_to(GraphInPtr graph, NodeVec ancestors) const {
                return std::make_shared<MakeConstant>(graph, ancestors[0]);
            }

            nodeType get_node_type() const {
                return CONSTANT_DERIVED;
            };

            Node get_parent_grad(Node my_grad, unsigned short index) {
                auto err = WrongGradient(NodeVec{owner, my_grad}, name);
                logger()->error() << name << "] " << err.msg;
                throw err;
            }
        };
    };

    namespace core {

        Node GraphInternal::constant_value(af::array value) {
            std::shared_ptr<Operator> op = std::make_shared<op::ConstantInput>(this, value);
            return derived_node(op);
        }

        Node GraphInternal::constant_value(double value, Shape shape) {
            std::shared_ptr<Operator> op = std::make_shared<op::ConstantValue>(this, max_float);
            return derived_node(op);
        }

        Node GraphInternal::constant_value(float value, Shape shape) {
            std::shared_ptr<Operator> op;
            if(max_float == f64) {
                op = std::make_shared<op::ConstantValue>(this, f32);
            } else {
                op = std::make_shared<op::ConstantValue>(this, max_float);
            }
            return derived_node(op);
        }

        Node GraphInternal::constant_value(long value, Shape shape) {
            std::shared_ptr<Operator> op = std::make_shared<op::ConstantValue>(this, max_int);
            return derived_node(op);
        }

        Node GraphInternal::constant_value(int value, Shape shape) {
            std::shared_ptr<Operator> op;
            if(max_int == i64) {
                op = std::make_shared<op::ConstantValue>(this, i32);
            } else {
                op = std::make_shared<op::ConstantValue>(this, max_int);
            }
            return derived_node(op);
        }

        Node GraphInternal::constant_value(short value, Shape shape) {
            std::shared_ptr<Operator> op;
            if(max_int == i64 or max_int == i32) {
                op = std::make_shared<op::ConstantValue>(this, i16);
            } else {
                op = std::make_shared<op::ConstantValue>(this, max_int);
            }
            return derived_node(op);
        }

        Node GraphInternal::constant_value(bool value, Shape shape) {
            std::shared_ptr<Operator> op = std::make_shared<op::ConstantValue>(this, b8);
            return derived_node(op);
        }

        Node GraphInternal::zeros(Shape shape, dType type) {
            return derived_node(std::make_shared<op::ConstantValue>(this, shape, 0, type));
        }

        Node GraphInternal::zeros(Shape shape) {
            return derived_node(std::make_shared<op::ConstantValue>(this, shape, 0, max_float));
        }

        Node GraphInternal::ones(Shape shape, dType type) {
            return derived_node(std::make_shared<op::ConstantValue>(this, shape, 1, type));
        }

        Node GraphInternal::ones(Shape shape) {
            return derived_node(std::make_shared<op::ConstantValue>(this, shape, 1, max_float));
        }

        Node GraphInternal::eye(SymInt size, dType type) {
            return derived_node(std::make_shared<op::Eye>(this, size, type));
        }

        Node GraphInternal::eye(SymInt size) {
            return derived_node(std::make_shared<op::Eye>(this, size, max_float));
        }

        Node GraphInternal::seq(SymInt start, SymInt end, dType type) {
            return derived_node(std::make_shared<op::Sequence>(this, start, end, type));
        }

        Node GraphInternal::seq(SymInt start, SymInt end) {
            return derived_node(std::make_shared<op::Sequence>(this, start, end, max_int));
        }

        Node Node::as_constant() {
            return apply<op::MakeConstant>(this);
        }
    }
}

#endif //METADIFF_OPERATORS_CONSTANTS_H
