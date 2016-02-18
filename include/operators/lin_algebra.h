//
// Created by alex on 16/12/15.
//

#ifndef METADIFF_OPERATORS_LIN_ALGEBRA_H
#define METADIFF_OPERATORS_LIN_ALGEBRA_H

namespace metadiff{

    /**
     * Inverts the order of all non singular dimensions (numpy)
     */
    class Transpose: public UnaryOperator{
    public:
        Transpose(GraphInPtr graph, Node parent) :
                UnaryOperator("Transpose", graph, parent) {}

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Transpose>(graph, ancestors[0]);
        }

        Shape get_shape() const{
            auto parent_shape = parent.unwrap()->shape;
            Shape shape {1, 1, 1, 1};
            int last_non_zero = 0;
            for(int i=3;i>=0;i--){
                if(parent_shape[i] != 1){
                    last_non_zero = i;
                    break;
                }
            }
            for(int i=0;i<=last_non_zero;i++){
                shape[i] = parent_shape[last_non_zero-i];
            }
            return shape;
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad.transpose();
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(parent.unwrap()->op->name == name){
                std::shared_ptr<Operator> base_op = parent.unwrap()->op->get_parents()[0].unwrap()->op;
                return base_op->equals(op) or op->equals(base_op);
            }
            if(name == op->name){
                std::shared_ptr<Transpose> cast_op = std::static_pointer_cast<Transpose>(op);
                return symbolic_equals(parent, cast_op->parent);
            }
            return false;
        }
    };

    Node Node::transpose() {
        // TODO a.transpose().transpose() = a
        return apply<Transpose>(this);
    }

    Node transpose(Node node){
        return node.transpose();
    }

    /**
     * General Matrix-Matrix Multiplication (GEMM)
     */
    class MatrixMultiplication: public NaryOperator{
    public:
        MatrixMultiplication(GraphInPtr graph,
                             NodeVec parents) :
                NaryOperator("MatrixMul", graph, parents)
        {
            if(not parents[0].is_matrix()){
                throw InvalidArguments(name, parents, "Parent 0 is not a matrix.");
            }
            for(int i=1;i<parents.size(); i++){
                if(not parents[i].is_matrix()){
                    throw InvalidArguments(name, parents, "Parent " + std::to_string(i) + " is not a matrix.");
                }
                if(parents[i-1].unwrap()->shape[1] != parents[i].unwrap()->shape[0]){
                    throw IncompatibleShapes(name, parents);
                }
            }
            shape = Shape{parents[0].unwrap()->shape[0], parents.back().unwrap()->shape[1], 1, 1};
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, NodeVec ancestors) const{
            return std::make_shared<MatrixMultiplication>(graph, ancestors);
        }

        MatrixMultiplication(GraphInPtr graph,
                             Node parent1,
                             Node parent2) :
        MatrixMultiplication(graph, {parent1, parent2})
        {}

        Node get_parent_grad(Node my_grad, size_t index){
            std::vector<Node> left_nodes;
            std::vector<Node> right_nodes;
            for (size_t p = 0; p < index; p++) {
                left_nodes.push_back(parents[p]);
            }
            for (size_t p = index + 1; p < parents.size(); p++) {
                right_nodes.push_back(parents[p]);
            }
            Node left_tr = Node();
            Node right_tr = Node();
            if (left_nodes.size() == 1) {
                left_tr = left_nodes[0].transpose();
            } else if (left_nodes.size() > 1) {
                left_tr = apply<MatrixMultiplication>(left_nodes);
                left_tr = left_tr.transpose();
            }

            if (right_nodes.size() == 1) {
                right_tr = right_nodes[0].transpose();
            } else if (right_nodes.size() > 1) {
                right_tr = apply<MatrixMultiplication>(right_nodes);
                right_tr = right_tr.transpose();
            }

            std::shared_ptr<Operator> op;
            if (left_tr.ptr.expired()) {
                return apply<MatrixMultiplication>(my_grad, right_tr);
            } else if (right_tr.ptr.expired()) {
                return apply<MatrixMultiplication>(left_tr, my_grad);
            } else {
                return apply<MatrixMultiplication>(NodeVec{left_tr, my_grad, right_tr});
            }
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name) {
                if (parents.size() != op->get_parents().size()) {
                    return false;
                }
                for (int i = 0; i < parents.size(); i++) {
                    if(not symbolic_equals(parents[i], op->get_parents()[i])){
                        return false;
                    }
                }
                return true;
            }
            return false;
        }
    };

    Node dot(NodeVec nodes){
        // TODO a dot a.inv() = eye
        // TODO a.transpose() */dot b.transpose() = (a */dot b).transpose()
        return apply<MatrixMultiplication>(nodes);
    };

    Node dot(Node node1, Node node2){
        return dot({node1, node2});
    }

    /**
     * Takes the matrix inverse
     */
    class MatrixInverse: public UnaryOperator{
    public:
        MatrixInverse(GraphInPtr graph, Node parent) :
                UnaryOperator("MatrixInv", graph, parent)
        {
            Shape parent_shape = parent.unwrap()->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw InvalidArguments(name, {parent}, "Parent must be a square matrix.");
            }
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<MatrixInverse>(graph, ancestors[0]);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node this_tr = owner.transpose();
            return dot(NodeVec{this_tr, my_grad, this_tr}).neg();
        }
    };

    Node Node::minv() {
        // TODO a.minv().minv() = a
        return apply<MatrixInverse>(this);
    }

    Node minv(Node node){
        return node.minv();
    }

    /**
     * Takes the determinant of a square matrix
     */
    class Determinant: public UnaryOperator{
    public:
        Determinant(GraphInPtr graph, Node parent) :
                UnaryOperator("Det", graph, parent)
        {
            auto parent_shape = parent.unwrap()->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw InvalidArguments(name, {parent}, "Parent must be a square matrix.");
            }
            if(parent.unwrap()->v_type == BOOLEAN){
                throw InvalidArguments(name, {parent}, "Parent can not be a BOOLEAN");
            }
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Determinant>(graph, ancestors[0]);
        }

        Shape get_shape() const{
            return {1, 1, 1, 1};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return mul(NodeVec{my_grad, owner, parent.minv().transpose()});
        }

    };

    Node Node::det() {
        return apply<Determinant>(this);
    }

    Node det(Node node){
        return node.det();
    }

    /**
     * Takes the log of the determinant of a square matrix
     */
    class LogDeterminant: public UnaryOperator{
    public:
        LogDeterminant(GraphInPtr graph, Node parent) :
                UnaryOperator("LogDet", graph, parent)
        {
            auto parent_shape = parent.unwrap()->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw InvalidArguments(name, {parent}, "Parent must be a square matrix.");
            }
            if(parent.unwrap()->v_type == BOOLEAN){
                throw InvalidArguments(name, {parent}, "Parent can not be a BOOLEAN");
            }
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<LogDeterminant>(graph, ancestors[0]);
        }

        Shape get_shape() const{
            return {1, 1, 1, 1};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return mul(NodeVec{my_grad, parent.minv().transpose()});
        }

    };

    Node Node::logdet() {
        return apply<LogDeterminant>(this);
    }

    Node logdet(Node node){
        return node.logdet();
    }

    /**
     * Takes the trace of a square matrix
     */
    class Trace: public UnaryOperator{
    public:
        Trace(GraphInPtr graph, Node parent) :
                UnaryOperator("Trace", graph, parent)
        {
            auto parent_shape = parent.unwrap()->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw InvalidArguments(name, {parent}, "Parent must be a square matrix.");
            }
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Trace>(graph, ancestors[0]);
        }

        Shape get_shape() const{
            return {1, 1, 1, 1};
        }

        ad_value_type get_value_type() const{
            if(parent.unwrap()->v_type == BOOLEAN){
                return INTEGER;
            } else {
                return parent.unwrap()->v_type;
            }
        };

        Node get_parent_grad(Node my_grad, size_t index){
            Node eye = graph->eye(parent.unwrap()->shape[0]);
            eye.unwrap()->grad_level = my_grad.unwrap()->grad_level;
            return mul(my_grad, eye);
        }
    };

    Node Node::trace(){
        return apply<Trace>(this);
    }

    Node trace(Node node){
        return node.logdet();
    }
}
#endif //METADIFF_OPERATORS_LIN_ALGEBRA_H
