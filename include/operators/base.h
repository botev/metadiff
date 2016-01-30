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
        Shape max_shape  = node_ptrs[0].unwrap()->shape;
        for(int i=1; i<node_ptrs.size();i++){
            Shape node_shape = node_ptrs[i].unwrap()->shape;
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

        NodeVec get_parents() const{
            return parents;
        };

        ad_value_type get_value_type() const{
            auto top_type = BOOLEAN;
            for(int i=0;i<parents.size();i++){
                auto v_type = parents[i].unwrap()->v_type;
                if(v_type == FLOAT){
                    return FLOAT;
                }
                if(v_type == INTEGER){
                    top_type = INTEGER;
                }
            }
            return top_type;
        };

        ad_node_type get_node_type() const{
            bool constant_derived = false;
            for(int i=0;i<parents.size();i++){
                if(parents[i].unwrap()->type == INPUT
                   or parents[i].unwrap()->type == INPUT_DERIVED
                   or parents[i].unwrap()->type == SHARED_INPUT){
                    return INPUT_DERIVED;
                }
                if(parents[i].unwrap()->type == CONSTANT_DERIVED or parents[i].unwrap()->type == SYMBOLIC_INTEGER){
                    constant_derived = true;
                }
            }
            if(constant_derived){
                return CONSTANT_DERIVED;
            } else {
                return CONSTANT;
            }
        };

        std::array<SymInt,4> get_shape() const{
            return shape;
        }

        size_t get_gradient_level() const{
            size_t max_grad_level = 0;
            for(int i=0;i<parents.size();i++){
                if(parents[i].unwrap()->grad_level > max_grad_level){
                    max_grad_level = parents[i].unwrap()->grad_level;
                }
            }
            return max_grad_level;
        };

        NodeVec get_arguments() const{
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

        NodeVec get_parents() const{
            return {parent1, parent2};
        };

        ad_value_type get_value_type() const{
            if(parent1.unwrap()->v_type == FLOAT or parent2.unwrap()->v_type == FLOAT){
                return FLOAT;
            } else if(parent1.unwrap()->v_type == INTEGER or parent2.unwrap()->v_type == INTEGER) {
                return INTEGER;
            } else {
                return BOOLEAN;
            }
        };

        ad_node_type get_node_type() const{
            if(parent1.unwrap()->type == INPUT
               or parent1.unwrap()->type == SHARED_INPUT
               or parent1.unwrap()->type == INPUT_DERIVED
               or parent2.unwrap()->type == INPUT
               or parent2.unwrap()->type == SHARED_INPUT
               or parent2.unwrap()->type == INPUT_DERIVED){
                return INPUT_DERIVED;
            }
            if(parent1.unwrap()->type == CONSTANT_DERIVED
               or parent1.unwrap()->type == SYMBOLIC_INTEGER
               or parent2.unwrap()->type == CONSTANT_DERIVED
               or parent2.unwrap()->type == SYMBOLIC_INTEGER){
                return CONSTANT_DERIVED;
            } else {
                return CONSTANT;
            }
        };

        std::array<SymInt,4> get_shape() const{
            return shape;
        }

        size_t get_gradient_level() const{
            return parent1.unwrap()->grad_level > parent2.unwrap()->grad_level ? parent1.unwrap()->grad_level : parent2.unwrap()->grad_level;
        };

        NodeVec get_arguments() const{
            return NodeVec {};
        }

        void throw_grad_type_error() const{
            throw UnknownError({parent1, parent2},
                               "Gradient message present, but parents are " +
                               to_string(parent1.unwrap()->type) + ", " +
                               to_string(parent2.unwrap()->type));
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<BinaryOperator> cast_op = std::static_pointer_cast<BinaryOperator>(op);
                return symbolic_equals(parent1, cast_op->parent1) and
                       symbolic_equals(parent2, cast_op->parent2);
            }
            return false;
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

        NodeVec get_parents() const{
            return {parent};
        };

        ad_value_type get_value_type() const{
            return parent.unwrap()->v_type;
        };

        ad_node_type get_node_type() const{
            if(parent.unwrap()->type == INPUT
               or parent.unwrap()->type == SHARED_INPUT
               or parent.unwrap()->type == INPUT_DERIVED){
                return INPUT_DERIVED;
            } else if (parent.unwrap()->type == CONSTANT_DERIVED or parent.unwrap()->type == SYMBOLIC_INTEGER){
                return CONSTANT_DERIVED;
            } else {
                return CONSTANT;
            }
        };

        Shape get_shape() const{
            return parent.unwrap()->shape;
        }

        size_t get_gradient_level() const{
            return parent.unwrap()->grad_level;
        };

        NodeVec get_arguments() const{
            return NodeVec {};
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<UnaryOperator> cast_op = std::static_pointer_cast<UnaryOperator>(op);
                return symbolic_equals(parent, cast_op->parent);
            }
            return false;
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
                if(parents[i].unwrap()->shape == shape or parents[i].is_scalar()){
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
                if(parents[i].unwrap()->shape == shape or parents[i].is_scalar()){
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

    class Alias : public UnaryOperator{
    public:
        Alias(GraphInPtr graph, Node parent):
                UnaryOperator("Alias", graph, parent) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Alias>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            std::shared_ptr<Operator> my_op = get_base_op(parent.unwrap()->op);
            return my_op->equals(op) or op->equals(my_op);
        }
    };

    Node Node::alias() {
        return apply<Alias>(unwrap());
    }

    Node alias(Node node){
        return apply<Alias>(node);
    }

    class Broadcast : public UnaryOperator {
    public:
        Shape to_shape;
        Broadcast(GraphInPtr graph,
                  Node parent,
                  Shape to_shape):
                UnaryOperator("Broadcast", graph, parent),
                to_shape(to_shape){
            for(int i=0;i<4;i++){
                if(parent.unwrap()->shape[i] != 1 and parent.unwrap()->shape[i] != to_shape[i]){
                    throw IncompatibleShapes(name, {parent.unwrap()->id}, {parent.unwrap()->shape, to_shape});
                }
            }
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Broadcast>(graph, ancestors[0], to_shape);
        }

        Shape get_shape() const{
            return to_shape;
        }

        std::vector<size_t> get_broadcast_axes() const{
            std::vector<size_t> axes;
            auto p1_shape = this->parent.unwrap()->shape;
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

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<Broadcast> cast_op = std::static_pointer_cast<Broadcast>(op);
                return symbolic_equals(parent, cast_op->parent) and to_shape == cast_op->to_shape;
            }
            return false;
        }
    };

    Node Node::broadcast(Shape shape) {
        std::shared_ptr<NodeInternal> ptr = unwrap();
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

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Sum>(graph, ancestors[0], axes);
        }

        Shape get_shape() const{
            Shape p_shape = parent.unwrap()->shape;
            for(int i=0;i<axes.size();i++){
                p_shape[axes[i]] = 1;
            }
            return p_shape;
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad.broadcast(parent.unwrap()->shape);
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<Sum> cast_op = std::static_pointer_cast<Sum>(op);
                return symbolic_equals(parent, cast_op->parent) and axes == cast_op->axes;
            }
            return false;
        }

    };

    Node Node::sum(std::vector<size_t> axes) {
        std::shared_ptr<NodeInternal> ptr = unwrap();
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

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Add>(graph, ancestors);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name) {
                bool check[parents.size()];
                for(int i=0;i<parents.size();i++){
                    check[i] = false;
                }
                if (parents.size() != op->get_parents().size()) {
                    return false;
                }
                for (int i = 0; i < parents.size(); i++) {
                    Node parent = op->get_parents()[i];
                    int j = 0;
                    for (; j < parents.size(); j++) {
                        if (symbolic_equals(parent, parents[j]) and not check[j]) {
                            check[j] = true;
                            break;
                        }
                    }
                    if (j == parents.size()) {
                        return false;
                    }
                }
            }
            return false;
        }
    };

    Node add(NodeVec nodes){
        // TODO a + (-a) = 0
        // TODO a * b + c * b = (a + c) * b ???
        std::vector<size_t> neg_indexes;
        for(size_t i=0;i<nodes.size();i++){
            if(nodes[i].unwrap()->op->name == "Neg"){
                neg_indexes.push_back(i);
            }
        }
        if(neg_indexes.size() == 0 or neg_indexes.size() == nodes.size()){
            return apply<Add>(nodes);
        } else {
            NodeVec reordered;
            for(size_t i=0;i<nodes.size();i++){
                if (std::find(neg_indexes.begin(), neg_indexes.end(), i) == neg_indexes.end()) {
                    reordered.push_back(nodes[i]);
                }
            }
            for(size_t i=0;i<neg_indexes.size();i++){
                reordered.push_back(nodes[neg_indexes[i]]);
            }
            return apply<Add>(reordered);
        }
    };

    Node add(Node node1, Node node2){
        return add({node1, node2});
    };

    Node operator+(Node node1, Node node2){
        return add({node1, node2});
    };

    void Operator::send_grad_message(size_t target, Node msg,
                                     std::vector<Node>& messages)  const{
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

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Neg>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad.neg();
        };
    };

    Node Node::neg(){
        // TODO x.neg().neg() = x
        return apply<Neg>(unwrap());
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

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Mul>(graph, ancestors);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            if(parents.size() == 2){
                // Special case when only two parents
                if(my_grad.unwrap()->op->name == "Ones"){
                    return parents[1-index];
                } else if(my_grad.unwrap()->op->name == "Zeros"){
                    return my_grad;
                }
                if(parents[1 - index].unwrap()->op->name == "Ones"){
                    return my_grad;
                } else if (parents[1-index].unwrap()->op->name == "Zeros"){
                    return parents[1-index];
                }
                return apply<Mul>(my_grad, parents[1 - index]);
            } else {
                Node product = apply<Mul>(my_grad, owner);
                return apply<Mul>(product, parents[index].div());
            }
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name) {
                bool check[parents.size()];
                for(int i=0;i<parents.size();i++){
                    check[i] = false;
                }
                if (parents.size() != op->get_parents().size()) {
                    return false;
                }
                for (int i = 0; i < parents.size(); i++) {
                    Node parent = op->get_parents()[i];
                    int j = 0;
                    for (; j < parents.size(); j++) {
                        if (symbolic_equals(parent, parents[j]) and not check[j]) {
                            check[j] = true;
                            break;
                        }
                    }
                    if (j == parents.size()) {
                        return false;
                    }
                }
            }
            return false;
        }
    };

    Node mul(NodeVec nodes){
        // TODO e^x * e^y = e^(x+y)
        // TODO x * x = x.square()
        // TODO x * (y / x) = y
        // Reorder so that Div operators are always at the end
        std::vector<size_t> div_indexes;
        for(size_t i=0;i<nodes.size();i++){
            if(nodes[i].unwrap()->op->name == "Div"){
                div_indexes.push_back(i);
            }
        }
        if(div_indexes.size() == 0 or div_indexes.size() == nodes.size()){
            return apply<Mul>(nodes);
        } else {
            NodeVec reordered;
            for(size_t i=0;i<nodes.size();i++){
                if (std::find(div_indexes.begin(), div_indexes.end(), i) == div_indexes.end()) {
                    reordered.push_back(nodes[i]);
                }
            }
            for(size_t i=0;i<div_indexes.size();i++){
                reordered.push_back(nodes[div_indexes[i]]);
            }
            return apply<Mul>(reordered);
        }

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

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Div>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return mul(my_grad, parent.square().div()).neg();
        }
    };

    Node Node::div() {
        // TODO x.div().div() = x
        return apply<Div>(unwrap());
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
                UnaryOperator("Square", graph, parent) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Square>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node two = graph->constant_value(2.0);
            two.unwrap()->grad_level = my_grad.unwrap()->grad_level;
            return mul({my_grad, two, parent});
        }
    };

    Node Node::square(){
        return apply<Square>(unwrap());
    }

    Node square(Node node){
        return node.square();
    }
}

#endif //METADIFF_OPERATORS_BASE_H
