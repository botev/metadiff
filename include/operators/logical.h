//
// Created by alex on 15/12/15.
//

#ifndef AUTODIFF_LOGICAL_H
#define AUTODIFF_LOGICAL_H
namespace autodiff {
    class LogicalBinary : public Operator{
    public:
        NodeInPtr parent1;
        NodeInPtr parent2;
        Shape shape;
        LogicalBinary(std::string const name,
                         GraphInPtr graph,
                         NodeInPtr parent1,
                      NodeInPtr parent2):
                Operator(graph, name),
                parent1(parent1),
                parent2(parent2)
        {
            //TODO verify shapes
        };

        NodeInVec get_parents() {
            return {parent1, parent2};
        };

        ad_value_type get_value_type(){
            return BOOLEAN;
        };

        ad_node_type get_node_type(){
            auto parent1_type = parent1.lock()->type;
            auto parent2_type = parent2.lock()->type;
            if(parent1_type == CONSTANT and parent2_type == CONSTANT){
                return CONSTANT;
            } else {
                return CONSTANT_DERIVED;
            }
        };

        std::array<SymInt,4> get_shape(){
            return shape;
        }

        unsigned short get_gradient_level(){
            auto parent1_grad_level = parent1.lock()->grad_level;
            auto parent2_grad_level = parent2.lock()->grad_level;
            return parent1_grad_level > parent2_grad_level ? parent1_grad_level : parent2_grad_level;
        };

        NodeInVec get_arguments() {
            return NodeInVec {};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();
            if (messages.find(current) != messages.end()) {
                throw UnkownError({parent1, parent2}, "The logical operator recieved a gradient message.");
            }
            return;
        };
    };

    class GreaterThan : public LogicalBinary {
    public:
        GreaterThan(GraphInPtr graph,
                    NodeInPtr parent1,
                    NodeInPtr parent2) :
                LogicalBinary("Gt", graph, parent1, parent2)
        {};
    };

    class GreaterThanOrEqual : public LogicalBinary {
    public:
        GreaterThanOrEqual(GraphInPtr graph,
                    NodeInPtr parent1,
                    NodeInPtr parent2) :
                LogicalBinary("Gte", graph, parent1, parent2)
        {};
    };

    class LessThan : public LogicalBinary {
    public:
        LessThan(GraphInPtr graph,
                    NodeInPtr parent1,
                    NodeInPtr parent2) :
                LogicalBinary("Lt", graph, parent1, parent2)
        {};
    };

    class LessThanOrEqual : public LogicalBinary {
    public:
        LessThanOrEqual(GraphInPtr graph,
                    NodeInPtr parent1,
                    NodeInPtr parent2) :
                LogicalBinary("Lte", graph, parent1, parent2)
        {};
    };

    class Equals : public LogicalBinary {
    public:
        Equals(GraphInPtr graph,
                    NodeInPtr parent1,
                    NodeInPtr parent2) :
                LogicalBinary("Eq", graph, parent1, parent2)
        {};
    };

    class NotEquals : public LogicalBinary {
    public:
        NotEquals(GraphInPtr graph,
                    NodeInPtr parent1,
                    NodeInPtr parent2) :
                LogicalBinary("Neq", graph, parent1, parent2)
        {};
    };

    class ApproximatelyEquals : public LogicalBinary {
    public:
        double tol;
        ApproximatelyEquals(GraphInPtr graph,
               NodeInPtr parent1,
               NodeInPtr parent2,
               double tol) :
                LogicalBinary("ApproxEq", graph, parent1, parent2),
                tol(tol)
        {};
    };

    class ApproximatelyNotEquals : public LogicalBinary {
    public:
        double tol;
        ApproximatelyNotEquals(GraphInPtr graph,
                           NodeInPtr parent1,
                           NodeInPtr parent2,
                           double tol) :
                LogicalBinary("ApproxNeq", graph, parent1, parent2),
                tol(tol)
        {};
    };
}
#endif //AUTODIFF_LOGICAL_H
