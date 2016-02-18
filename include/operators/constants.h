//
// Created by alex on 17/12/15.
//

#ifndef METADIFF_OPERATORS_CONSTANTS_H
#define METADIFF_OPERATORS_CONSTANTS_H

namespace metadiff{
    /**
     * Operator which is just a view of the parent, but is always constant
     * meaning no gradients will be passed
     */
    class MakeConstant: public UnaryOperator{
    public:
        MakeConstant(GraphInPtr graph,
                     Node parent):
                UnaryOperator("MakeConst", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<MakeConstant>(graph, ancestors[0]);
        }

        ad_node_type get_node_type() const{
            if(parent.unwrap()->type == CONSTANT){
                return CONSTANT;
            } else {
                return CONSTANT_DERIVED;
            }
        };

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }
    };

    Node Node::constant() {
        return apply<MakeConstant>(this);
    }

    /**
     * Abstract class for any operators which produce constant expressions
     */
    class ConstantOperator: public Operator{
    public:
        Shape shape;
        ConstantOperator(std::string const name,
                         GraphInPtr graph):
                Operator(name, graph) {};

        ConstantOperator(std::string const name,
                         GraphInPtr graph,
                         Shape shape):
                Operator(name, graph), shape(shape) {};

        NodeVec get_parents() const{
            return {};
        };

        ad_value_type get_value_type() const{
            return FLOAT;
        };

        ad_node_type get_node_type() const{
            for(int i=0;i<4;i++){
                if(not shape[i].is_constant()){
                    return CONSTANT_DERIVED;
                }
            }
            return CONSTANT;
        };

        Shape get_shape() const{
            return shape;
        }

        size_t get_gradient_level() const{
            return 0;
        };

        NodeVec get_arguments() const{
            return NodeVec {};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<ConstantOperator> cast_op = std::static_pointer_cast<ConstantOperator>(op);
                return shape == cast_op->shape;
            }
            return false;
        }
    };

    /**
     * Matrix identity
     */
    class Eye: public ConstantOperator{
    public:
        Eye(GraphInPtr graph, SymInt size):
                ConstantOperator("Eye", graph)
        {
            shape = {size, size, 1, 1};
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Eye>(graph, shape[0]);
        }

    };

    Node GraphInternal::eye(SymInt size) {
        return derived_node(std::make_shared<Eye>(this, size));
    }

    /**
     * Matrix filled with zeros
     */
    class Zeros: public ConstantOperator{
    public:
        Zeros(GraphInPtr graph, Shape shape):
                ConstantOperator("Zeros", graph, shape){};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Zeros>(graph, shape);
        }

    };

    Node GraphInternal::zeros(Shape shape) {
        return derived_node(std::make_shared<Zeros>(this, shape));
    }

    /**
     * Matrix filled with ones
     */
    class Ones: public ConstantOperator{
    public:
        Ones(GraphInPtr graph, Shape shape):
                ConstantOperator("Ones", graph, shape) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Ones>(graph, shape);
        }

    };

    Node GraphInternal::ones(Shape shape) {
        return derived_node(std::make_shared<Ones>(this, shape));
    }

    /**
     * Matrix filled with the same value
     */
    class ConstantValue: public ConstantOperator{
    public:
        double value;
        ConstantValue(GraphInPtr graph, Shape shape, double value):
                ConstantOperator("Value", graph, shape),
                value(value) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<ConstantValue>(graph, shape, value);
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<ConstantValue> cast_op = std::static_pointer_cast<ConstantValue>(op);
                return shape == cast_op->shape and value == cast_op->value;
            } else if(op->name == "Ones"){
                std::shared_ptr<Ones> cast_op = std::static_pointer_cast<Ones>(op);
                return shape == cast_op->shape and value == 1.0;
            } else if(op->name == "Zeros"){
                std::shared_ptr<Zeros> cast_op = std::static_pointer_cast<Zeros>(op);
                return shape == cast_op->shape and value == 0.0;
            }
            return false;
        }
    };

    Node GraphInternal::constant_value(double value, Shape shape){
        if(value == 0.0){
            return zeros(shape);
        } else if(value == 1.0){
            return ones(shape);
        } else{
            return derived_node(std::make_shared<ConstantValue>(this, shape, value));
        }
    }

//    double Operator::get_scalar_value()  const{
//        if(name == "Zeros") {
//            return 0;
//        }
//        if(name == "Ones"){
//            return 1;
//        }
//        if(name == "Value"){
//            return dynamic_cast<const ConstantValue* const>(this)->value;
//        }
//        return owner.unwrap()->value.host<float>()[0];
//    }

    /**
     * A vector of the sequence from 'start' to 'end'
     */
    class Sequence: public ConstantOperator{
    public:
        SymInt start;
        SymInt end;
        Sequence(GraphInPtr graph, SymInt start, SymInt end):
                ConstantOperator("Sequence", graph),
                start(start),
                end(end) {
            shape = Shape {end - start, 1, 1, 1};
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Sequence>(graph, start, end);
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<Sequence> cast_op = std::static_pointer_cast<Sequence>(op);
                return start == cast_op->start and end == cast_op->end;
            }
            return false;
        }
    };

    Node GraphInternal::seq(SymInt start, SymInt end) {
        return derived_node(std::make_shared<Sequence>(this, start, end));
    }
}

#endif //METADIFF_OPERATORS_CONSTANTS_H
