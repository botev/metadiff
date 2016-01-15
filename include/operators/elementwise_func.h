//
// Created by alex on 15/12/15.
//

#ifndef METADIFF_ELEMENTWISE_FUNC_H
#define METADIFF_ELEMENTWISE_FUNC_H
namespace metadiff {

    class Exp: public UnaryOperator {
    public:
        Exp(GraphInPtr graph, Node parent) :
                UnaryOperator("Exp", graph, parent)
        {};


        Node get_parent_grad(Node my_grad, size_t index){
            return mul(my_grad, owner);
        }
    };

    Node NodeInternal::exp() {
        return apply<Exp>();
    }

    Node exp(Node node){
        return node->exp();
    }

    class Log: public UnaryOperator {
    public:
        Log(GraphInPtr graph, Node parent) :
                UnaryOperator("Log", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node div = parent->div();
            div->update_grad_level();
            return mul(my_grad, div);
        }
    };

    Node NodeInternal::log() {
        return apply<Log>();
    }

    Node log(Node node){
        return node->log();
    }

    class Pow : public ElementwiseBinary {
    public:
        Pow(GraphInPtr graph, Node parent1, Node parent2) :
                ElementwiseBinary("Pow", graph, parent1, parent2) { };

        Node get_parent_grad(Node my_grad, size_t index){
            Node product = mul(my_grad, owner);
            product->update_grad_level();
            if(index == 0){
                Node div = parent1->div();
                div->update_grad_level();
                return mul(NodeVec{product, parent2, div});
            } else {
                return mul(NodeVec{product, parent1->log()});
            }
        }
    };

    Node NodeInternal::pow(Node power) {
        return graph->derived_node(std::make_shared<Pow>(graph, this, power));
    }

    Node NodeInternal::pow(double d_power) {
        Node power = graph->value(d_power);
        return graph->derived_node(std::make_shared<Pow>(graph, this, power));
    }

    Node pow(Node node, Node power){
        return node->pow(power);
    }

    Node pow(double value, Node power){
        Node node = power->graph->value(value);
        return power->graph->derived_node(std::make_shared<Pow>(power->graph, node, power));
    }

    class Abs: public UnaryOperator {
    public:
        Abs(GraphInPtr graph, Node parent) :
                UnaryOperator("Abs", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node zero = graph->value(0.0);
            zero->grad_level = my_grad->grad_level;
            Node ge = parent->ge(zero);
            ge->update_grad_level();
            return mul(my_grad, ge);
        }
    };

    Node NodeInternal::abs() {
        return apply<Abs>();
    }

    Node abs(Node node){
        return node->abs();
    }

    class Sigmoid: public UnaryOperator {
    public:
        Sigmoid(GraphInPtr graph, Node parent) :
                UnaryOperator("Sigmoid", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node one = graph->value(1.0);
            one->grad_level = my_grad->grad_level;
            Node neg = owner->neg();
            neg->update_grad_level();
            return mul({my_grad, owner, add(one, owner->neg())});
        }
    };

    Node NodeInternal::sigmoid() {
        return apply<Sigmoid>();
    }

    Node sigmoid(Node node){
        return node->sigmoid();
    }

    class Softplus : public UnaryOperator{
    public:
        double threshold;
        Softplus(GraphInPtr graph,
                 Node parent,
                 double threshold = 50):
                UnaryOperator("Softplus", graph, parent),
                threshold(threshold)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node sigmoid = parent->sigmoid();
            sigmoid->update_grad_level();
            return mul({my_grad, sigmoid});
        }
    };

    Node NodeInternal::softplus(double threshold) {
        return graph->derived_node(std::make_shared<Softplus>(graph, this, threshold));
    }

    Node softplus(Node node, double threshold = 50){
        return node->softplus(threshold);
    }

    class Sin: public UnaryOperator {
    public:
        Sin(GraphInPtr graph, Node parent) :
                UnaryOperator("Sin", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node cos = parent->cos();
            cos->update_grad_level();
            return mul(my_grad, cos);
        }
    };

    Node NodeInternal::sin() {
        return apply<Sin>();
    }

    Node sin(Node node){
        return node->sin();
    }

    class Cos: public UnaryOperator {
    public:
        Cos(GraphInPtr graph, Node parent) :
                UnaryOperator("Cos", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node sin = parent->cos();
            sin->update_grad_level();
            return mul(my_grad, sin->neg());
        }
    };

    Node NodeInternal::cos(){
        return apply<Cos>();
    }

    Node cos(Node node){
        return node->cos();
    }

    class Tan: public UnaryOperator {
    public:
        Tan(GraphInPtr graph, Node parent) :
                UnaryOperator("Tan", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node cos = parent->cos();
            cos->update_grad_level();
            return mul(my_grad, cos->square()->div());
        }
    };

    Node NodeInternal::tan() {
        return apply<Tan>();
    }

    Node tan(Node node){
        return node->tan();
    }

    class Cot: public UnaryOperator {
    public:
        Cot(GraphInPtr graph, Node parent) :
                UnaryOperator("Cot", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node sin = parent->cos();
            sin->update_grad_level();
            return mul(my_grad, sin->square()->neg())->neg();
        }
    };

    Node NodeInternal::cot() {
        return apply<Cot>();
    }

    Node cot(Node node){
        return node->cot();
    }

    class Sinh: public UnaryOperator {
    public:
        Sinh(GraphInPtr graph, Node parent) :
                UnaryOperator("Sinh", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node cosh = parent->cosh();
            cosh->update_grad_level();
            return mul(my_grad, cosh);
        }
    };

    Node NodeInternal::sinh() {
        return apply<Sinh>();
    }

    Node sinh(Node node){
        return node->sinh();
    }

    class Cosh: public UnaryOperator {
    public:
        Cosh(GraphInPtr graph, Node parent) :
                UnaryOperator("Cosh", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node sinh = parent->sinh();
            sinh->update_grad_level();
            return mul(my_grad, sinh);
        }
    };

    Node NodeInternal::cosh() {
        return apply<Cosh>();
    }

    Node cosh(Node node){
        return node->cosh();
    }

    class Tanh: public UnaryOperator {
    public:
        Tanh(GraphInPtr graph, Node parent) :
                UnaryOperator("Tanh", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node one = graph->value(1.0);
            one->grad_level = my_grad->grad_level;
            Node square = owner->square();
            square->update_grad_level();
            return mul(my_grad, add(one, square->neg()));
        }
    };

    Node NodeInternal::tanh() {
        return apply<Tanh>();
    }

    Node tanh(Node node){
        return node->tanh();
    }

    class Coth: public UnaryOperator {
    public:
        Coth(GraphInPtr graph, Node parent) :
                UnaryOperator("Coth", graph, parent)
        {};

        Node get_parent_grad(Node my_grad, size_t index){
            Node one = graph->value(1.0);
            one->grad_level = my_grad->grad_level;
            Node square = owner->square();
            square->update_grad_level();
            return mul(my_grad, add(one, square->neg()));
        }
    };

    Node NodeInternal::coth() {
        return apply<Coth>();
    }

    Node coth(Node node){
        return node->coth();
    }
}

#endif //METADIFF_ELEMENTWISE_FUNC_H
