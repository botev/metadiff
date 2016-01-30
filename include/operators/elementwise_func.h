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

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Exp>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return mul(my_grad, owner);
        }
    };

    Node Node::exp() {
        return apply<Exp>(this);
    }

    Node exp(Node node){
        return node.exp();
    }

    class Log: public UnaryOperator {
    public:
        Log(GraphInPtr graph, Node parent) :
                UnaryOperator("Log", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Log>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node div = parent.div();
            return mul(my_grad, div);
        }
    };

    Node Node::log() {
        return apply<Log>(this);
    }

    Node log(Node node){
        return node.log();
    }

    class Pow : public ElementwiseBinary {
    public:
        Pow(GraphInPtr graph, Node parent1, Node parent2) :
                ElementwiseBinary("Pow", graph, parent1, parent2) { };

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Pow>(graph, ancestors[0], ancestors[1]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node product = mul(my_grad, owner);
            if(index == 0){
                Node div = parent1.div();
                return mul(NodeVec{product, parent2, div});
            } else {
                return mul(NodeVec{product, parent1.log()});
            }
        }
    };

    Node Node::pow(Node power) {
        std::shared_ptr<NodeInternal> ptr = unwrap();
        return ptr->graph->derived_node(std::make_shared<Pow>(ptr->graph, this, power));
    }

    Node Node::pow(double d_power) {
        std::shared_ptr<NodeInternal> ptr = unwrap();
        Node power = ptr->graph->constant_value(d_power);
        return ptr->graph->derived_node(std::make_shared<Pow>(ptr->graph, this, power));
    }

    Node pow(Node node, Node power){
        return node.pow(power);
    }

    Node pow(double value, Node power){
        Node node = power.unwrap()->graph->constant_value(value);
        return power.unwrap()->graph->derived_node(std::make_shared<Pow>(power.unwrap()->graph, node, power));
    }

    class Abs: public UnaryOperator {
    public:
        Abs(GraphInPtr graph, Node parent) :
                UnaryOperator("Abs", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Abs>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node zero = graph->constant_value(0.0);
            zero.unwrap()->grad_level = my_grad.unwrap()->grad_level;
            Node ge = parent.ge(zero);
            return mul(my_grad, ge);
        }
    };

    Node Node::abs() {
        return apply<Abs>(this);
    }

    Node abs(Node node){
        return node.abs();
    }

    Node Node::sigmoid() {
        std::shared_ptr<NodeInternal> ptr = unwrap();
        return ptr->graph->constant_value(1.0) / (ptr->graph->constant_value(1.0) + neg().exp());
    }

    Node sigmoid(Node node){
        return node.unwrap()->graph->constant_value(1.0) / (node.unwrap()->graph->constant_value(1.0) + node.neg().exp());
    }

    class Log1p : public UnaryOperator{
    public:
        Log1p(GraphInPtr graph,
                 Node parent):
                UnaryOperator("Log1p", graph, parent) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Log1p>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node sigmoid = parent.sigmoid();
            return mul({my_grad, sigmoid});
        }
    };

    Node Node::log1p(){
        return apply<Log1p>(this);
    }

    Node log1p(Node node){
        return apply<Log1p>(node);
    }

    Node Node::softplus(size_t threshold) {
        std::shared_ptr<NodeInternal> ptr = unwrap();
        return (this > ptr->graph->constant_value(threshold)).select(this, exp().log1p());
    }

    Node softplus(Node node, size_t threshold = 50){
        return select(node > node.unwrap()->graph->constant_value(threshold), node, node.exp().log1p());
    }

    class Sin: public UnaryOperator {
    public:
        Sin(GraphInPtr graph, Node parent) :
                UnaryOperator("Sin", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Sin>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node cos = parent.cos();
            return mul(my_grad, cos);
        }
    };

    Node Node::sin() {
        return apply<Sin>(this);
    }

    Node sin(Node node){
        return node.sin();
    }

    class Cos: public UnaryOperator {
    public:
        Cos(GraphInPtr graph, Node parent) :
                UnaryOperator("Cos", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Cos>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node sin = parent.cos();
            return mul(my_grad, sin.neg());
        }
    };

    Node Node::cos(){
        return apply<Cos>(this);
    }

    Node cos(Node node){
        return node.cos();
    }

    class Tan: public UnaryOperator {
    public:
        Tan(GraphInPtr graph, Node parent) :
                UnaryOperator("Tan", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Tan>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node cos = parent.cos();
            return mul(my_grad, cos.square().div());
        }
    };

    Node Node::tan() {
        return apply<Tan>(this);
    }

    Node tan(Node node){
        return node.tan();
    }

    class Cot: public UnaryOperator {
    public:
        Cot(GraphInPtr graph, Node parent) :
                UnaryOperator("Cot", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Cot>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node sin = parent.cos();
            return mul(my_grad, sin.square().neg()).neg();
        }
    };

    Node Node::cot() {
        return apply<Cot>(this);
    }

    Node cot(Node node){
        return node.cot();
    }

    class Sinh: public UnaryOperator {
    public:
        Sinh(GraphInPtr graph, Node parent) :
                UnaryOperator("Sinh", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Sinh>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node cosh = parent.cosh();
            return mul(my_grad, cosh);
        }
    };

    Node Node::sinh() {
        return apply<Sinh>(this);
    }

    Node sinh(Node node){
        return node.sinh();
    }

    class Cosh: public UnaryOperator {
    public:
        Cosh(GraphInPtr graph, Node parent) :
                UnaryOperator("Cosh", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Cosh>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node sinh = parent.sinh();
            return mul(my_grad, sinh);
        }
    };

    Node Node::cosh() {
        return apply<Cosh>(this);
    }

    Node cosh(Node node){
        return node.cosh();
    }

    class Tanh: public UnaryOperator {
    public:
        Tanh(GraphInPtr graph, Node parent) :
                UnaryOperator("Tanh", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Tanh>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node one = graph->constant_value(1.0);
            one.unwrap()->grad_level = my_grad.unwrap()->grad_level;
            Node square = owner.square();
            return mul(my_grad, add(one, square.neg()));
        }
    };

    Node Node::tanh() {
        return apply<Tanh>(this);
    }

    Node tanh(Node node){
        return node.tanh();
    }

    class Coth: public UnaryOperator {
    public:
        Coth(GraphInPtr graph, Node parent) :
                UnaryOperator("Coth", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Coth>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node one = graph->constant_value(1.0);
            one.unwrap()->grad_level = my_grad.unwrap()->grad_level;
            Node square = owner.square();
            return mul(my_grad, add(one, square.neg()));
        }
    };

    Node Node::coth() {
        return apply<Coth>(this);
    }

    Node coth(Node node){
        return node.coth();
    }
}

#endif //METADIFF_ELEMENTWISE_FUNC_H
