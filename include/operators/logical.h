//
// Created by alex on 15/12/15.
//

#ifndef METADIFF_OPERATORS_LOGICAL_H
#define METADIFF_OPERATORS_LOGICAL_H

namespace metadiff {

    class LogicalBinary : public ElementwiseBinary{
    public:
        LogicalBinary(std::string const name,
                      GraphInPtr graph,
                      Node parent1,
                      Node parent2):
                ElementwiseBinary(name, graph, parent1, parent2)
        {};

        ad_value_type get_value_type(){
            return BOOLEAN;
        };

        ad_node_type get_node_type(){
            if(parent1->type == CONSTANT and parent2->type == CONSTANT){
                return CONSTANT;
            } else {
                return CONSTANT_DERIVED;
            }
        };

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }
    };

    class LogicalUnary : public UnaryOperator{
    public:
        LogicalUnary(std::string const name,
                     GraphInPtr graph,
                     Node parent):
                UnaryOperator(name, graph, parent)
        {};

        ad_value_type get_value_type(){
            return BOOLEAN;
        };

        ad_node_type get_node_type(){
            if(parent->type == CONSTANT){
                return CONSTANT;
            } else {
                return CONSTANT_DERIVED;
            }
        };

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }
    };

    class GreaterThan : public LogicalBinary {
    public:
        GreaterThan(GraphInPtr graph,
                    Node parent1,
                    Node parent2) :
                LogicalBinary("Gt", graph, parent1, parent2)
        {};
    };

    Node NodeInternal::gt(Node node) {
        return apply<GreaterThan>(this, node);
    }

    Node gt(Node node1, Node node2){
        return apply<GreaterThan>(node1, node2);
    }

    Node operator>(Node node1, Node node2){
        return apply<GreaterThan>(node1, node2);
    }

    class GreaterThanOrEqual : public LogicalBinary {
    public:
        GreaterThanOrEqual(GraphInPtr graph,
                           Node parent1,
                           Node parent2) :
                LogicalBinary("Ge", graph, parent1, parent2)
        {};
    };

    Node NodeInternal::ge(Node node) {
        return apply<GreaterThanOrEqual>(this, node);
    }

    Node ge(Node node1, Node node2){
        return apply<GreaterThanOrEqual>(node1, node2);
    }

    Node operator>=(Node node1, Node node2){
        return apply<GreaterThanOrEqual>(node1, node2);
    }

    class LessThan : public LogicalBinary {
    public:
        LessThan(GraphInPtr graph,
                 Node parent1,
                 Node parent2) :
                LogicalBinary("Lt", graph, parent1, parent2)
        {};
    };

    Node NodeInternal::lt(Node node) {
        return apply<LessThan>(this, node);
    }

    Node lt(Node node1, Node node2){
        return apply<LessThan>(node1, node2);
    }

    Node operator<(Node node1, Node node2){
        return apply<LessThan>(node1, node2);
    }

    class LessThanOrEqual : public LogicalBinary {
    public:
        LessThanOrEqual(GraphInPtr graph,
                        Node parent1,
                        Node parent2) :
                LogicalBinary("Le", graph, parent1, parent2)
        {};
    };

    Node NodeInternal::le(Node node) {
        return apply<LessThanOrEqual>(this, node);
    }

    Node le(Node node1, Node node2){
        return apply<LessThanOrEqual>(node1, node2);
    }

    Node operator<=(Node node1, Node node2){
        return apply<LessThanOrEqual>(node1, node2);
    }

    class Equals : public LogicalBinary {
    public:
        Equals(GraphInPtr graph,
               Node parent1,
               Node parent2) :
                LogicalBinary("Eq", graph, parent1, parent2)
        {};
    };

    Node NodeInternal::eq(Node node){
        return apply<Equals>(this, node);
    }

    Node eq(Node node1, Node node2){
        return apply<Equals>(node1, node2);
    }

    Node operator==(Node node1, Node node2){
        return apply<Equals>(node1, node2);
    }

    class NotEquals : public LogicalBinary {
    public:
        NotEquals(GraphInPtr graph,
                  Node parent1,
                  Node parent2) :
                LogicalBinary("Ne", graph, parent1, parent2)
        {};
    };

    Node NodeInternal::neq(Node node){
        return apply<NotEquals>(this, node);
    }

    Node neq(Node node1, Node node2){
        return apply<NotEquals>(node1, node2);
    }

    Node operator!=(Node node1, Node node2){
        return apply<NotEquals>(node1, node2);
    }

    class ApproximatelyEquals : public LogicalBinary {
    public:
        double tol;
        ApproximatelyEquals(GraphInPtr graph,
                            Node parent1,
                            Node parent2,
                            double tol) :
                LogicalBinary("ApproxEq", graph, parent1, parent2),
                tol(tol)
        {};
    };

    Node NodeInternal::approx_eq(Node node){
        return apply<ApproximatelyEquals>(this, node);
    }

    Node approx_eq(Node node1, Node node2){
        return apply<ApproximatelyEquals>(node1, node2);
    }

    class ApproximatelyNotEquals : public LogicalBinary {
    public:
        double tol;
        ApproximatelyNotEquals(GraphInPtr graph,
                               Node parent1,
                               Node parent2,
                               double tol) :
                LogicalBinary("ApproxNe", graph, parent1, parent2),
                tol(tol)
        {};
    };

    Node NodeInternal::approx_neq(Node node){
        return apply<ApproximatelyNotEquals>(this, node);
    }

    Node approx_neq(Node node1, Node node2){
        return apply<ApproximatelyNotEquals>(node1, node2);
    }

    class And : public LogicalBinary {
    public:
        And(GraphInPtr graph,
            Node parent1,
            Node parent2) :
                LogicalBinary("And", graph, parent1, parent2)
        {
            if(parent1->v_type != BOOLEAN or parent2->v_type != BOOLEAN){
                throw UnknownError({parent1, parent2}, "Operator 'And' accepts only BOOLEAN inputs");
            }
        };
    };

    Node NodeInternal::logical_and(Node node){
        return apply<And>(this, node);
    }

    Node logical_and(Node node1, Node node2){
        return apply<And>(node1, node2);
    }

    Node operator&&(Node node1, Node node2){
        return apply<And>(node1, node2);
    }

    class Or : public LogicalBinary {
    public:
        Or(GraphInPtr graph,
           Node parent1,
           Node parent2) :
                LogicalBinary("Or", graph, parent1, parent2)
        {
            if(parent1->v_type != BOOLEAN or parent2->v_type != BOOLEAN){
                throw UnknownError({parent1, parent2}, "Operator 'Or' accepts only BOOLEAN inputs");
            }
        };
    };

    Node NodeInternal::logical_or(Node node){
        return apply<Or>(this, node);
    }

    Node logical_or(Node node1, Node node2){
        return apply<Or>(node1, node2);
    }

    Node operator||(Node node1, Node node2){
        return apply<Or>(node1, node2);
    }

    class ZeroElements : public LogicalUnary {
    public:
        ZeroElements(GraphInPtr graph,
                     Node parent) :
                LogicalUnary("ZeroElem", graph, parent)
        {};
    };

    Node NodeInternal::zero_elem() {
        return apply<ZeroElements>();
    }

    Node zero_elem(Node node){
        return node->apply<ZeroElements>();
    }

    class NonZeroElements : public LogicalUnary {
    public:
        NonZeroElements(GraphInPtr graph,
                        Node parent) :
                LogicalUnary("NonZeroElem", graph, parent)
        {};
    };

    Node NodeInternal::non_zero_elem() {
        return apply<NonZeroElements>();
    }

    class IsNaN : public LogicalUnary {
    public:
        IsNaN(GraphInPtr graph,
              Node parent) :
                LogicalUnary("IsNaN", graph, parent)
        {};
    };

    Node NodeInternal::is_nan(){
        return apply<IsNaN>();
    }

    Node is_nan(Node node){
        return node->is_nan();
    }

    class IsInf : public LogicalUnary {
    public:
        IsInf(GraphInPtr graph,
              Node parent) :
                LogicalUnary("IsInf", graph, parent)
        {};
    };

    Node NodeInternal::is_inf(){
        return apply<IsInf>();
    }

    Node is_inf(Node node){
        return node->is_inf();
    }

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
            if(condition->v_type != BOOLEAN){
                throw InvalidArguments(name, {condition, trueParent, falseParent},
                                       "The condition must have a value type BOOLEAN");
            }
            if(trueParent->v_type != falseParent->v_type){
                throw InvalidArguments(name, {condition, trueParent, falseParent},
                                       "The true and false statement must be of the same value type");
            }
            if(trueParent->is_constant()){
                this->parent1 = parent1->broadcast(shape);
            } else if(falseParent->is_constant()){
                this->parent1 = parent2->broadcast(shape);
            }
        };

        NodeVec get_arguments() {
            return NodeVec {condition};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node zero = graph->value(0.0);
            zero->grad_level = my_grad->grad_level;
            if(index == 0){
                return condition->select(my_grad, zero);
            } else {
                return condition->select(zero, my_grad);
            }
        };
    };

    Node NodeInternal::select(Node result_true, Node result_false){
        return  graph->derived_node(std::make_shared<Select>(graph, this, result_true, result_false));
    }

    Node select(Node condition, Node result_true, Node result_false){
        return  condition->select(result_true, result_false);
    }
}
#endif //METADIFF_OPERATORS_LOGICAL_H
