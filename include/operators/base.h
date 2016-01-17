//
// Created by alex on 13/12/15.
//

#ifndef METADIFF_OPERATORS_BASE_H
#define METADIFF_OPERATORS_BASE_H

namespace metadiff {

    // Helper function to validate axes
    bool validate_axes(std::vector<size_t> axes){
        if(axes.size() > 4){
            return false;
        }
        bool checks[4] {false, false, false, false};
        for(int i=0;i<axes.size();i++){
            if(axes[i] > 3){
                return false;
            }
            if(checks[axes[i]]){
                return false;
            }
            checks[axes[i]] = true;
        }
        return true;
    }

    // Helper function to verify shapes of elementwise operators
    Shape verify_elementwise_shapes(std::string name, NodeVec node_ptrs){
        Shape max_shape  = node_ptrs[0].ptr->shape;
        for(int i=1; i<node_ptrs.size();i++){
            Shape node_shape = node_ptrs[i].ptr->shape;
            bool max = false;
            for(int j=0;j<4;j++){
                if(node_shape[j] != 1 and max_shape[j] == 1){
                    max = true;
                    break;
                }
            }
            if(max){
                for(int j=0;j<4;j++) {
                    if(node_shape[j] == 1 and max_shape[j] != 1){
                        throw IncompatibleShapes(name, node_ptrs);
                    } else if(node_shape[j] != 1 and max_shape[j] != 1 and node_shape[j] != max_shape[j]){
                        throw IncompatibleShapes(name, node_ptrs);
                    }
                }
                max_shape = node_shape;
            }
        }
        return max_shape;
    }

    class NaryOperator: public Operator{
    public:
        NodeVec parents;
        Shape shape;
        NaryOperator(std::string const name,
                     GraphInPtr graph,
                     NodeVec parents) :
                Operator(name, graph),
                parents(parents)
        {
            if(parents.size() < 2){
                throw InvalidArguments(name, parents, "Need atleast 2 parents");
            }
        };

        NodeVec get_parents() {
            return parents;
        };

        ad_value_type get_value_type(){
            auto top_type = BOOLEAN;
            for(int i=0;i<parents.size();i++){
                auto v_type = parents[i].ptr->v_type;
                if(v_type == FLOAT){
                    return FLOAT;
                }
                if(v_type == INTEGER){
                    top_type = INTEGER;
                }
            }
            return top_type;
        };

        ad_node_type get_node_type(){
            bool constant_derived = false;
            for(int i=0;i<parents.size();i++){
                if(parents[i].ptr->type == INPUT
                   or parents[i].ptr->type == INPUT_DERIVED
                   or parents[i].ptr->type == SHARED_INPUT){
                    return INPUT_DERIVED;
                }
                if(parents[i].ptr->type == CONSTANT_DERIVED or parents[i].ptr->type == SYMBOLIC_INTEGER){
                    constant_derived = true;
                }
            }
            if(constant_derived){
                return CONSTANT_DERIVED;
            } else {
                return CONSTANT;
            }
        };

        std::array<SymInt,4> get_shape(){
            return shape;
        }

        size_t get_gradient_level(){
            size_t max_grad_level = 0;
            for(int i=0;i<parents.size();i++){
                if(parents[i].ptr->grad_level > max_grad_level){
                    max_grad_level = parents[i].ptr->grad_level;
                }
            }
            return max_grad_level;
        };

        NodeVec get_arguments() {
            return NodeVec {};
        }
    };

    class BinaryOperator : public Operator{
    public:
        Node parent1;
        Node parent2;
        Shape shape;

        BinaryOperator(std::string const name,
                       GraphInPtr graph,
                       Node parent1,
                       Node parent2) :
                Operator(name, graph),
                parent1(parent1),
                parent2(parent2)
        {}

        NodeVec get_parents() {
            return {parent1, parent2};
        };

        ad_value_type get_value_type(){
            if(parent1.ptr->v_type == FLOAT or parent2.ptr->v_type == FLOAT){
                return FLOAT;
            } else if(parent1.ptr->v_type == INTEGER or parent2.ptr->v_type == INTEGER) {
                return INTEGER;
            } else {
                return BOOLEAN;
            }
        };

        ad_node_type get_node_type(){
            if(parent1.ptr->type == INPUT
               or parent1.ptr->type == SHARED_INPUT
               or parent1.ptr->type == INPUT_DERIVED
               or parent2.ptr->type == INPUT
               or parent2.ptr->type == SHARED_INPUT
               or parent2.ptr->type == INPUT_DERIVED){
                return INPUT_DERIVED;
            }
            if(parent1.ptr->type == CONSTANT_DERIVED
               or parent1.ptr->type == SYMBOLIC_INTEGER
               or parent2.ptr->type == CONSTANT_DERIVED
               or parent2.ptr->type == SYMBOLIC_INTEGER){
                return CONSTANT_DERIVED;
            } else {
                return CONSTANT;
            }
        };

        std::array<SymInt,4> get_shape(){
            return shape;
        }

        size_t get_gradient_level(){
            return parent1.ptr->grad_level > parent2.ptr->grad_level ? parent1.ptr->grad_level : parent2.ptr->grad_level;
        };

        NodeVec get_arguments() {
            return NodeVec {};
        }

        void throw_grad_type_error(){
            throw UnknownError({parent1, parent2},
                               "Gradient message present, but parents are " +
                               to_string(parent1.ptr->type) + ", " +
                               to_string(parent2.ptr->type));
        }
    };

    class UnaryOperator : public Operator{
    public:
        Node parent;
        UnaryOperator(std::string const name,
                      GraphInPtr graph,
                      Node parent):
                Operator(name, graph),
                parent(parent)
        {};

        NodeVec get_parents() {
            return {parent};
        };

        ad_value_type get_value_type(){
            return parent.ptr->v_type;
        };

        ad_node_type get_node_type(){
            if(parent.ptr->type == INPUT
               or parent.ptr->type == SHARED_INPUT
               or parent.ptr->type == INPUT_DERIVED){
                return INPUT_DERIVED;
            } else if (parent.ptr->type == CONSTANT_DERIVED or parent.ptr->type == SYMBOLIC_INTEGER){
                return CONSTANT_DERIVED;
            } else {
                return CONSTANT;
            }
        };

        Shape get_shape(){
            return parent.ptr->shape;
        }

        size_t get_gradient_level(){
            return parent.ptr->grad_level;
        };

        NodeVec get_arguments() {
            return NodeVec {};
        }
    };

    class ElementwiseNary : public NaryOperator{
    public:
        ElementwiseNary(std::string const name,
                        GraphInPtr graph,
                        NodeVec parents) :
                NaryOperator(name, graph, parents){
            this->parents.clear();
            shape = verify_elementwise_shapes(name, parents);
            for(int i=0;i<parents.size();i++){
                if(parents[i].ptr->shape == shape or parents[i].is_scalar()){
                    this->parents.push_back(parents[i]);
                } else if(graph->broadcast == ad_implicit_broadcast::RAISE){
                    throw ImplicitBroadcast(name, parents);
                } else{
                    if(graph->broadcast == ad_implicit_broadcast::WARN){
                        auto msg = ImplicitBroadcast(name, parents);
                        std::cout << "WARNING:" << msg.get_message() << std::endl;
                    }
                    this->parents.push_back(parents[i].broadcast(shape));
                }
            }
        };
    };

    class ElementwiseBinary : public BinaryOperator{
    public:
        ElementwiseBinary(std::string const name,
                          GraphInPtr graph,
                          Node parent1,
                          Node parent2) :
                BinaryOperator(name, graph, parent1, parent2) {
            NodeVec parents = get_parents();
            shape = verify_elementwise_shapes(name, {parents});
            for(int i=0;i<2;i++){
                if(parents[i].ptr->shape == shape or parents[i].is_scalar()){
                    continue;
                } else if(graph->broadcast == ad_implicit_broadcast::RAISE){
                    throw ImplicitBroadcast(name, parents);
                } else{
                    if(graph->broadcast == ad_implicit_broadcast::WARN){
                        auto msg = ImplicitBroadcast(name, parents);
                        std::cout << "WARNING:" << msg.get_message() << std::endl;
                    }
                    if(i == 0){
                        this->parent1 = parent1.broadcast(shape);
                    } else {
                        this->parent2 = parent2.broadcast(shape);
                    }
                }
            }
        }
    };

    class Broadcast : public UnaryOperator {
    public:
        Shape to_shape;
        Broadcast(GraphInPtr graph,
                  Node parent,
                  Shape to_shape):
                UnaryOperator("Broadcast", graph, parent),
                to_shape(to_shape){
            for(int i=0;i<4;i++){
                if(parent.ptr->shape[i] != 1 and parent.ptr->shape[i] != to_shape[i]){
                    throw IncompatibleShapes(name, {parent.ptr->id}, {parent.ptr->shape, to_shape});
                }
            }
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<Broadcast>(graph, ancestors[0], to_shape);
        }

        Shape get_shape(){
            return to_shape;
        }

        std::vector<size_t> get_broadcast_axes(){
            std::vector<size_t> axes;
            auto p1_shape = this->parent.ptr->shape;
            for(size_t i=0;i<4;i++){
                if(p1_shape[i] != to_shape[i]){
                    axes.push_back(i);
                }
            }
            return axes;
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad.sum(get_broadcast_axes());
        }
    };

    Node Node::broadcast(Shape shape) {
        return ptr->graph->derived_node(std::make_shared<Broadcast>(ptr->graph, this, shape));
    }

    
    class Sum : public UnaryOperator {
    public:
        std::vector<size_t> axes;

        Sum(GraphInPtr graph,
            Node parent,
            std::vector<size_t> axes):
                UnaryOperator("Sum", graph, parent),
                axes(axes)
        {
            if(not validate_axes(axes)){
                std::string axes_str;
                for(int i=0;i<axes.size();i++){
                    axes_str += std::to_string(axes[i]);
                    if(i < axes.size()-1){
                        axes_str += ", ";
                    }
                }
                if(axes.size() == 0){
                    axes_str = "NULL";
                }
                throw InvalidArguments(name, {parent}, axes_str);
            }
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<Sum>(graph, ancestors[0], axes);
        }

        Shape get_shape(){
            Shape p_shape = parent.ptr->shape;
            for(int i=0;i<axes.size();i++){
                p_shape[axes[i]] = 1;
            }
            return p_shape;
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad.broadcast(parent.ptr->shape);
        }

    };

    Node Node::sum(std::vector<size_t> axes) {
        return ptr->graph->derived_node(std::make_shared<Sum>(ptr->graph, this, axes));
    }

    class Add : public ElementwiseNary {
    public:
        Add(GraphInPtr graph, NodeVec parents) :
                ElementwiseNary("Add", graph, parents)
        {}

        Add(GraphInPtr graph, Node parent1, Node parent2) :
                Add(graph, {parent1, parent2})
        {}

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<Add>(graph, ancestors);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }
    };

    Node add(std::vector<Node> nodes){
        auto graph = nodes[0].ptr->graph;
        return graph->derived_node(std::make_shared<Add>(graph, nodes));
    };

    Node add(Node node1, Node node2){
        return add({node1, node2});
    };

    Node operator+(Node node1, Node node2){
        return add({node1, node2});
    };

    void Operator::send_grad_message(size_t target, Node msg,
                                     std::vector<Node>& messages){
        if (not messages[target].empty()) {
            // If not first message add them and then send the sum
            messages[target] = add(messages[target], msg);
        } else {
            // If first just send it
            messages[target] = msg;
        }
    }

    class Neg : public UnaryOperator {
    public:
        Neg(GraphInPtr graph, Node parent) :
                UnaryOperator("Neg", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<Neg>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad.neg();
        };
    };

    Node Node::neg(){
        return apply<Neg>(this);
    }

    Node operator-(Node node){
        return node.neg();
    }

    Node operator-(Node node1, Node node2){
        return add(node1, node2.neg());
    }

    class Mul : public ElementwiseNary {
    public:
        Mul(GraphInPtr graph, NodeVec parents) :
                ElementwiseNary("Mul", graph, parents)
        {};

        Mul(GraphInPtr graph, Node p1, Node p2) :
                ElementwiseNary("Mul", graph, {p1, p2})
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<Mul>(graph, ancestors);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            if(parents.size() == 2){
                // Special case when only two parents
                if(my_grad.ptr->op->name == "Ones"){
                    return parents[1-index];
                } else if(my_grad.ptr->op->name == "Zeros"){
                    return my_grad;
                }
                if(parents[1 - index].ptr->op->name == "Ones"){
                    return my_grad;
                } else if (parents[1-index].ptr->op->name == "Zeros"){
                    return parents[1-index];
                }
                return apply<Mul>(my_grad, parents[1 - index]);
            } else {
                Node product = apply<Mul>(my_grad, owner);
                return apply<Mul>(product, parents[index].div());
            }
        }
    };

    Node mul(NodeVec nodes){
        return apply<Mul>(nodes);
    };

    Node mul(Node node1, Node node2){
        return mul({node1, node2});
    }

    Node operator*(Node node1, Node node2){
        return mul(NodeVec{node1, node2});
    };

    class Div : public UnaryOperator {
    public:
        Div(GraphInPtr graph, Node parent) :
                UnaryOperator("Div", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<Div>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return mul(my_grad, parent.square().div()).neg();
        }
    };

    Node Node::div() {
        return apply<Div>(this);
    }

    Node div(Node node1, Node node2){
        return mul(node1, node2.div());
    }

    Node operator/(Node node1, Node node2){
        return mul(node1, node2.div());
    };

    class Square : public UnaryOperator {
    public:
        Square(GraphInPtr graph, Node parent) :
                UnaryOperator("Square", graph, parent)
        {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<Square>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node two = graph->constant_value(2.0);
            two.ptr->grad_level = my_grad.ptr->grad_level;
            return mul({my_grad, two, parent});
        }
    };

    Node Node::square(){
        return apply<Square>(this);
    }

    Node square(Node node){
        return node.square();
    }
}

#endif //METADIFF_OPERATORS_BASE_H
