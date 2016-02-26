//
// Created by alex on 15/12/15.
//

#ifndef METADIFF_OPERATORS_LOGICAL_H
#define METADIFF_OPERATORS_LOGICAL_H

namespace metadiff {

    /**
     * Abstract class for any binary logical operators
     */
    class LogicalBinary : public ElementwiseBinary{
    public:
        LogicalBinary(std::string const name,
                      GraphInPtr graph,
                      Node parent1,
                      Node parent2):
                ElementwiseBinary(name, graph, parent1, parent2) {};

        ad_value_type get_value_type() const{
            return BOOLEAN;
        };

        ad_node_type get_node_type() const{
            if(parent1.unwrap()->type == CONSTANT and parent2.unwrap()->type == CONSTANT){
                return CONSTANT;
            } else {
                return CONSTANT_DERIVED;
            }
        };

        Node get_parent_grad(Node my_grad, size_t index){
            throw WrongGradient(name, {parent1, parent2});
        }
    };

    /**
     * Abstract class for any unary logical operators
     */
    class LogicalUnary : public UnaryOperator{
    public:
        LogicalUnary(std::string const name,
                     GraphInPtr graph,
                     Node parent):
                UnaryOperator(name, graph, parent) {};

        ad_value_type get_value_type() const{
            return BOOLEAN;
        };

        ad_node_type get_node_type() const{
            if(parent.unwrap()->type == CONSTANT){
                return CONSTANT;
            } else {
                return CONSTANT_DERIVED;
            }
        };

        Node get_parent_grad(Node my_grad, size_t index){
            throw WrongGradient(name, {parent});
        }
    };

    /**
     * Elementwise NOT operation
     */
    class Not : public LogicalUnary {
    public:
        Not(GraphInPtr graph,
                     Node parent) :
                LogicalUnary("Not", graph, parent) {
            if(parent.unwrap()->v_type != BOOLEAN){
                throw InvalidArguments(name, {parent},
                                       "The operator accepts only BOOLEAN inputs");
            }
        };

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Not>(graph, ancestors[0]);
        }
    };

    Node Node::nt() {
        return apply<Not>(this);
    }

    Node operator!(Node node){
        return apply<Not>(node);
    }

    /**
     * Elementwise comparison '>'
     */
    class GreaterThan : public LogicalBinary {
    public:
        GreaterThan(GraphInPtr graph,
                    Node parent1,
                    Node parent2) :
                LogicalBinary("Gt", graph, parent1, parent2) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<GreaterThan>(graph, ancestors[0], ancestors[1]);
        }
    };


    Node gt(Node node1, Node node2){
        return apply<GreaterThan>(node1, node2);
    }

    Node Node::gt(Node node) {
        return apply<GreaterThan>(this, node);
    }

    Node operator>(Node node1, Node node2){
        return apply<GreaterThan>(node1, node2);
    }

    /**
     * Elementwise comparison '>='
     */
    class GreaterThanOrEqual : public LogicalBinary {
    public:
        GreaterThanOrEqual(GraphInPtr graph,
                           Node parent1,
                           Node parent2) :
                LogicalBinary("Ge", graph, parent1, parent2) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<GreaterThanOrEqual>(graph, ancestors[0], ancestors[1]);
        }
    };

    Node Node::ge(Node node) {
        return apply<GreaterThanOrEqual>(this, node);
    }

    Node ge(Node node1, Node node2){
        return apply<GreaterThanOrEqual>(node1, node2);
    }

    Node operator>=(Node node1, Node node2){
        return apply<GreaterThanOrEqual>(node1, node2);
    }

    /**
     * Elementwise comparison '<'
     */
    class LessThan : public LogicalBinary {
    public:
        LessThan(GraphInPtr graph,
                 Node parent1,
                 Node parent2) :
                LogicalBinary("Lt", graph, parent1, parent2) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<LessThan>(graph, ancestors[0], ancestors[1]);
        }
    };

    Node Node::lt(Node node) {
        return apply<LessThan>(this, node);
    }

    Node lt(Node node1, Node node2){
        return apply<LessThan>(node1, node2);
    }

    Node operator<(Node node1, Node node2){
        return apply<LessThan>(node1, node2);
    }

    /**
     * Elementwise comparison '<='
     */
    class LessThanOrEqual : public LogicalBinary {
    public:
        LessThanOrEqual(GraphInPtr graph,
                        Node parent1,
                        Node parent2) :
                LogicalBinary("Le", graph, parent1, parent2) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<LessThanOrEqual>(graph, ancestors[0], ancestors[1]);
        }
    };

    Node Node::le(Node node) {
        return apply<LessThanOrEqual>(this, node);
    }

    Node le(Node node1, Node node2){
        return apply<LessThanOrEqual>(node1, node2);
    }

    Node operator<=(Node node1, Node node2){
        return apply<LessThanOrEqual>(node1, node2);
    }

    /**
     * Elementwise comparison '=='
     */
    class Equals : public LogicalBinary {
    public:
        Equals(GraphInPtr graph,
               Node parent1,
               Node parent2) :
                LogicalBinary("Eq", graph, parent1, parent2) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Equals>(graph, ancestors[0], ancestors[1]);
        }
    };

    Node Node::eq(Node node){
        return apply<Equals>(this, node);
    }

    Node eq(Node node1, Node node2){
        return apply<Equals>(node1, node2);
    }

    Node operator==(Node node1, Node node2){
        return apply<Equals>(node1, node2);
    }

    Node Node::neq(Node node){
        return !(this == node);
    }

    Node neq(Node node1, Node node2){
        return !(node1 == node2);
    }

    Node operator!=(Node node1, Node node2){
        return !(node1 == node2);
    }

    /**
     * Checks if the two nodes are approximately equal (up to a tolerance measure)
     * This is particulary useful for floating point nodes, where machine precision
     * might have effect.
     */
    class ApproximatelyEquals : public LogicalBinary {
    public:
        double tol;
        ApproximatelyEquals(GraphInPtr graph,
                            Node parent1,
                            Node parent2,
                            double tol) :
                LogicalBinary("ApproxEq", graph, parent1, parent2),
                tol(tol) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<ApproximatelyEquals>(graph, ancestors[0], ancestors[1], tol);
        }
    };

    Node Node::approx_eq(Node node, double tol){
        GraphInPtr graph = unwrap()->graph;
        return graph->derived_node(std::make_shared<ApproximatelyEquals>(graph, this, node, tol));
    }

    Node approx_eq(Node node1, Node node2, double tol = 0.00001){
        GraphInPtr graph = node1.unwrap()->graph;
        return graph->derived_node(std::make_shared<ApproximatelyEquals>(graph, node1, node2, tol));
    }

    Node Node::approx_neq(Node node, double tol){
        return !(this->approx_eq(node, tol));
    }

    Node approx_neq(Node node1, Node node2, double tol=0.00001){
        return !(node1.approx_eq(node2, tol));
    }

    /**
     * Elementwise logical AND
     */
    class And : public LogicalBinary {
    public:
        And(GraphInPtr graph,
            Node parent1,
            Node parent2) :
                LogicalBinary("And", graph, parent1, parent2)
        {
            if(parent1.unwrap()->v_type != BOOLEAN or parent2.unwrap()->v_type != BOOLEAN){
                throw InvalidArguments(name, {parent1, parent2}, "Operator 'And' accepts only BOOLEAN parents");
            }
        };

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<And>(graph, ancestors[0], ancestors[1]);
        }
    };

    Node Node::logical_and(Node node){
        return apply<And>(this, node);
    }

    Node logical_and(Node node1, Node node2){
        return apply<And>(node1, node2);
    }

    Node operator&&(Node node1, Node node2){
        return apply<And>(node1, node2);
    }

    /**
     * Elementwise logical OR
     */
    class Or : public LogicalBinary {
    public:
        Or(GraphInPtr graph,
           Node parent1,
           Node parent2) :
                LogicalBinary("Or", graph, parent1, parent2)
        {
            if(parent1.unwrap()->v_type != BOOLEAN or parent2.unwrap()->v_type != BOOLEAN){
                throw InvalidArguments(name, {parent1, parent2}, "Operator 'Or' accepts only BOOLEAN parents");
            }
        };

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Or>(graph, ancestors[0], ancestors[1]);
        }
    };

    Node Node::logical_or(Node node){
        return apply<Or>(this, node);
    }

    Node logical_or(Node node1, Node node2){
        return apply<Or>(node1, node2);
    }

    Node operator||(Node node1, Node node2){
        return apply<Or>(node1, node2);
    }

    /**
     * Checks every element if its equal to 0
     */
    class ZeroElements : public LogicalUnary {
    public:
        ZeroElements(GraphInPtr graph,
                     Node parent) :
                LogicalUnary("ZeroElem", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<ZeroElements>(graph, ancestors[0]);
        }
    };

    Node Node::zero_elem() {
        return apply<ZeroElements>(this);
    }

    Node zero_elem(Node node){
        return apply<ZeroElements>(node);
    }

    Node Node::non_zero_elem() {
        return !(this->zero_elem());
    }

    Node non_zero_elem(Node node){
        return !(node.zero_elem());
    }

    /**
     * Checks every element if its is NaN
     */
    class IsNaN : public LogicalUnary {
    public:
        IsNaN(GraphInPtr graph,
              Node parent) :
                LogicalUnary("IsNaN", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<IsNaN>(graph, ancestors[0]);
        }
    };

    Node Node::is_nan(){
        return apply<IsNaN>(this);
    }

    Node is_nan(Node node){
        return apply<IsNaN>(node);
    }

    /**
     * Checks every element if its is Inf
     */
    class IsInf : public LogicalUnary {
    public:
        IsInf(GraphInPtr graph,
              Node parent) :
                LogicalUnary("IsInf", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<IsInf>(graph, ancestors[0]);
        }
    };

    Node Node::is_inf(){
        return apply<IsInf>(this);
    }

    Node is_inf(Node node){
        return apply<IsInf>(node);
    }

    /**
     * Elementwise selects one of the two parents based on the condition
     * Both the parents and the condition node must be of the same size.
     */
    class Select: public BinaryOperator{
    public:
        Node condition;
        Select(GraphInPtr graph,
               Node condition,
               Node trueParent,
               Node falseParent):
                BinaryOperator("Select", graph, trueParent, falseParent),
                condition(condition)
        {
            shape = verify_elementwise_shapes(name, NodeVec{condition, trueParent, falseParent});
            if(condition.unwrap()->v_type != BOOLEAN){
                throw InvalidArguments(name, {condition, trueParent, falseParent},
                                       "The condition must have a value type BOOLEAN");
            }
            if(trueParent.unwrap()->v_type != falseParent.unwrap()->v_type){
                throw InvalidArguments(name, {condition, trueParent, falseParent},
                                       "The true and false statement must be of the same value type");
            }
            if(trueParent.is_constant()){
                this->parent1 = parent1.broadcast(shape);
            } else if(falseParent.is_constant()){
                this->parent1 = parent2.broadcast(shape);
            }
        };

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Select>(graph, ancestors[2], ancestors[0], ancestors[1]);
        }

        NodeVec get_arguments() const {
            return NodeVec {condition};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node zero = graph->constant_value(0.0);
            zero.unwrap()->grad_level = my_grad.unwrap()->grad_level;
            if(index == 0){
                return condition.select(my_grad, zero);
            } else {
                return condition.select(zero, my_grad);
            }
        };
    };

    Node Node::select(Node result_true, Node result_false){
        return unwrap()->graph->derived_node(std::make_shared<Select>(unwrap()->graph, this, result_true, result_false));
    }

    Node select(Node condition, Node result_true, Node result_false){
        return  condition.select(result_true, result_false);
    }
}
#endif //METADIFF_OPERATORS_LOGICAL_H
