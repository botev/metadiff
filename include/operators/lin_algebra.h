//
// Created by alex on 16/12/15.
//

#ifndef METADIFF_OPERATORS_LIN_ALGEBRA_H
#define METADIFF_OPERATORS_LIN_ALGEBRA_H

namespace metadiff{

    // Inverts the order of all non singular dimensions (numpy)
    class Transpose: public UnaryOperator{
    public:
        Transpose(GraphInPtr graph, Node parent) :
                UnaryOperator("Transpose", graph, parent)
        {}

        Shape get_shape(){
            auto parent_shape = parent.ptr->shape;
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
    };

    Node Node::transpose() {
        return apply<Transpose>(this);
    }

    Node transpose(Node node){
        return node.transpose();
    }

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
                if(parents[i-1].ptr->shape[1] != parents[i].ptr->shape[0]){
                    throw IncompatibleShapes(name, parents);
                }
            }
            shape = Shape{parents[0].ptr->shape[0], parents.back().ptr->shape[1], 1, 1};
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
                left_tr.update_grad_level();
            } else if (left_nodes.size() > 1) {
                left_tr = apply<MatrixMultiplication>(left_nodes);
                left_tr.update_grad_level();
                left_tr = left_tr.transpose();
            }

            if (right_nodes.size() == 1) {
                right_tr = right_nodes[0].transpose();
                right_tr.update_grad_level();
            } else if (right_nodes.size() > 1) {
                right_tr = apply<MatrixMultiplication>(right_nodes);
                right_tr.update_grad_level();
                right_tr = right_tr.transpose();
            }

            std::shared_ptr<Operator> op;
            if (left_tr.empty()) {
                return apply<MatrixMultiplication>(my_grad, right_tr);
            } else if (right_tr.empty()) {
                return apply<MatrixMultiplication>(left_tr, my_grad);
            } else {
                return apply<MatrixMultiplication>(NodeVec{left_tr, my_grad, right_tr});
            }
        }
    };

    Node dot(NodeVec nodes){
        return apply<MatrixMultiplication>(nodes);
    };

    Node dot(Node node1, Node node2){
        return dot({node1, node2});
    }

    class MatrixInverse: public UnaryOperator{
    public:
        MatrixInverse(GraphInPtr graph, Node parent) :
                UnaryOperator("MatrixInv", graph, parent)
        {
            Shape parent_shape = parent.ptr->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw UnknownError({parent}, "Operator 'MatrixInverse' takes only squared matrices");
            }
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node this_tr = owner.transpose();
            this_tr.update_grad_level();
            return dot(NodeVec{this_tr, my_grad, this_tr}).neg();
        }
    };

    Node Node::minv() {
        return apply<MatrixInverse>(this);
    }

    Node minv(Node node){
        return node.minv();
    }

    class Determinant: public UnaryOperator{
    public:
        Determinant(GraphInPtr graph, Node parent) :
                UnaryOperator("Det", graph, parent)
        {
            auto parent_shape = parent.ptr->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw UnknownError({parent}, "Operator 'Determinant' takes only squared matrices");
            }
            if(parent.ptr->v_type == BOOLEAN){
                throw UnknownError({parent}, "Operator 'Determinant' is not applicable for node of type BOOLEAN");
            }
        }

        Shape get_shape(){
            return {1, 1, 1, 1};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node inv = parent.minv();
            inv.update_grad_level();
            return mul(NodeVec{my_grad, owner, inv.transpose()});
        }
    };

    Node Node::det() {
        return apply<Determinant>(this);
    }

    Node det(Node node){
        return node.det();
    }

    class LogDeterminant: public UnaryOperator{
    public:
        LogDeterminant(GraphInPtr graph, Node parent) :
                UnaryOperator("LogDet", graph, parent)
        {
            auto parent_shape = parent.ptr->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw UnknownError({parent}, "Operator 'Determinant' takes only squared matrices");
            }
            if(parent.ptr->v_type == BOOLEAN){
                throw UnknownError({parent}, "Operator 'Determinant' is not applicable for node of type BOOLEAN");
            }
        }

        Shape get_shape(){
            return {1, 1, 1, 1};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            Node inv = parent.minv();
            inv.update_grad_level();
            return mul(NodeVec{my_grad, owner, inv});
        }
    };

    Node Node::logdet() {
        return apply<LogDeterminant>(this);
    }

    Node logdet(Node node){
        return node.logdet();
    }

    class Trace: public UnaryOperator{
    public:
        Trace(GraphInPtr graph, Node parent) :
                UnaryOperator("Trace", graph, parent)
        {
            auto parent_shape = parent.ptr->shape;
            if(parent_shape[0] != parent_shape[1] or parent_shape[2] != 1 or parent_shape[2] !=1){
                throw UnknownError({parent}, "Operator 'Trace' takes only squared matrices");
            }
        }

        Shape get_shape(){
            return {1, 1, 1, 1};
        }

        ad_value_type get_value_type(){
            if(parent.ptr->v_type == BOOLEAN){
                return INTEGER;
            } else {
                return parent.ptr->v_type;
            }
        };

        Node get_parent_grad(Node my_grad, size_t index){
            Node eye = graph->eye(parent.ptr->shape[0]);
            eye.ptr->grad_level = my_grad.ptr->grad_level;
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
