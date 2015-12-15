//
// Created by alex on 15/12/15.
//

#ifndef AUTODIFF_LOGICAL_H
#define AUTODIFF_LOGICAL_H
namespace metadiff {
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
            NodeInVec parents = get_parents();
            try{
                shape = verify_shapes({parents});
            } catch(const int){
                throw IncompatibleShapes(name, parents);
            }
            for(int i=0;i<2;i++){
                auto parent = parents[i].lock();
                if(parent->shape == shape or parent->is_scalar()){
                    continue;
                } else if(graph.lock()->broadcast == ad_implicit_broadcast::RAISE){
                    throw ImplicitBroadcast(name, parents);
                } else{
                    if(graph.lock()->broadcast == ad_implicit_broadcast::WARN){
                        auto msg = ImplicitBroadcast(name, parents);
                        std::cout << "WARNING:" << msg.get_message() << std::endl;
                    }
                    auto  op = std::make_shared<Broadcast>(this->graph, parents[i], shape);
                    if(i == 0){
                        this->parent1 = graph.lock()->derived_node(op);
                    } else {
                        this->parent2 = graph.lock()->derived_node(op);
                    }
                }
            }
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

    class LogicalUnary : public Operator{
    public:
        NodeInPtr parent;
        LogicalUnary(std::string const name,
                     GraphInPtr graph,
                     NodeInPtr parent):
                Operator(graph, name),
                parent(parent)
        {};

        NodeInVec get_parents() {
            return {parent};
        };

        ad_value_type get_value_type(){
            return BOOLEAN;
        };

        ad_node_type get_node_type(){
            auto parent_type = parent.lock()->type;
            if(parent_type == CONSTANT){
                return CONSTANT;
            } else {
                return CONSTANT_DERIVED;
            }
        };

        std::array<SymInt,4> get_shape(){
            return parent.lock()->shape;
        }

        unsigned short get_gradient_level(){
            return parent.lock()->grad_level;
        };

        NodeInVec get_arguments() {
            return NodeInVec {};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t> &messages) {
            auto graph = this->graph.lock();
            if (messages.find(current) != messages.end()) {
                throw UnkownError({parent}, "The logical operator recieved a gradient message.");
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

    Node gt(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<GreaterThan>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node operator>(Node node1, Node node2){
        return gt(node1, node2);
    }

    class GreaterThanOrEqual : public LogicalBinary {
    public:
        GreaterThanOrEqual(GraphInPtr graph,
                           NodeInPtr parent1,
                           NodeInPtr parent2) :
                LogicalBinary("Gte", graph, parent1, parent2)
        {};
    };

    Node gte(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<GreaterThanOrEqual>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node operator>=(Node node1, Node node2){
        return gte(node1, node2);
    }

    class LessThan : public LogicalBinary {
    public:
        LessThan(GraphInPtr graph,
                 NodeInPtr parent1,
                 NodeInPtr parent2) :
                LogicalBinary("Lt", graph, parent1, parent2)
        {};
    };

    Node lt(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<LessThan>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node operator<(Node node1, Node node2){
        return lt(node1, node2);
    }

    class LessThanOrEqual : public LogicalBinary {
    public:
        LessThanOrEqual(GraphInPtr graph,
                        NodeInPtr parent1,
                        NodeInPtr parent2) :
                LogicalBinary("Lte", graph, parent1, parent2)
        {};
    };

    Node lte(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<LessThanOrEqual>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node operator<=(Node node1, Node node2){
        return lte(node1, node2);
    }

    class Equals : public LogicalBinary {
    public:
        Equals(GraphInPtr graph,
               NodeInPtr parent1,
               NodeInPtr parent2) :
                LogicalBinary("Eq", graph, parent1, parent2)
        {};
    };

    Node eq(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<Equals>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node operator==(Node node1, Node node2){
        return eq(node1, node2);
    }

    class NotEquals : public LogicalBinary {
    public:
        NotEquals(GraphInPtr graph,
                  NodeInPtr parent1,
                  NodeInPtr parent2) :
                LogicalBinary("Neq", graph, parent1, parent2)
        {};
    };

    Node neq(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<NotEquals>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node operator!=(Node node1, Node node2){
        return neq(node1, node2);
    }

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

    Node approx_eq(Node node1, Node node2, double tol=1e-9){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<ApproximatelyEquals>(graph, arg1, arg2, tol);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

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

    Node approx_neq(Node node1, Node node2, double tol = 1e-9){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<ApproximatelyNotEquals>(graph, arg1, arg2, tol);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    class And : public LogicalBinary {
    public:
        And(GraphInPtr graph,
                  NodeInPtr parent1,
                  NodeInPtr parent2) :
                LogicalBinary("And", graph, parent1, parent2)
        {
            if(parent1.lock()->v_type != BOOLEAN or parent2.lock()->v_type != BOOLEAN){
                throw UnkownError({parent1, parent2}, "Operator 'And' accepts only BOOLEAN inputs");
            }
        };
    };

    Node logical_and(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<And>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node operator&&(Node node1, Node node2){
        return logical_and(node1, node2);
    }

    class Or : public LogicalBinary {
    public:
        Or(GraphInPtr graph,
            NodeInPtr parent1,
            NodeInPtr parent2) :
                LogicalBinary("Or", graph, parent1, parent2)
        {
            if(parent1.lock()->v_type != BOOLEAN or parent2.lock()->v_type != BOOLEAN){
                throw UnkownError({parent1, parent2}, "Operator 'Or' accepts only BOOLEAN inputs");
            }
        };
    };

    Node logical_or(Node node1, Node node2){
        auto graph = node1.graph.lock();
        auto arg1 = graph->nodes[node1.id];
        auto arg2 = graph->nodes[node2.id];
        auto op = std::make_shared<Or>(graph, arg1, arg2);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node operator||(Node node1, Node node2){
        return logical_or(node1, node2);
    }

    class ZeroElements : public LogicalUnary {
    public:
        ZeroElements(GraphInPtr graph,
                     NodeInPtr parent) :
                LogicalUnary("ZeroElem", graph, parent)
        {};
    };

    Node Node::zeros() {
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<ZeroElements>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node zero_elem(Node node){
        return node.zeros();
    }

    class NonZeroElements : public LogicalUnary {
    public:
        NonZeroElements(GraphInPtr graph,
                     NodeInPtr parent) :
                LogicalUnary("NonZeroElem", graph, parent)
        {};
    };

    Node Node::non_zeros(){
        auto graph = this->graph.lock();
        auto arg = graph->nodes[this->id];
        auto op = std::make_shared<NonZeroElements>(graph, arg);
        return Node(graph, graph->derived_node(op).lock()->id);
    }

    Node non_zero_elem(Node node){
        return node.non_zeros();
    }
}
#endif //AUTODIFF_LOGICAL_H
