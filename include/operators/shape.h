//
// Created by alex on 18/12/15.
//

#ifndef METADIFF_OPERATORS_SHAPE_H
#define METADIFF_OPERATORS_SHAPE_H

namespace metadiff{

    // Helper function to get all elements
    SymInt number_of_elements(Shape shape){
        return shape[0] * shape[1] * shape[2] * shape[3];
    };

    class Diagonal: public UnaryOperator{
    public:
        Shape shape;
        Diagonal(GraphInPtr graph, Node parent):
                UnaryOperator("Diag", graph, parent){
            if(not parent.is_matrix()){
                throw InvalidArguments(name, {parent}, "Parent is not a vector or a sqyare matrix.");
            }
            if(parent.is_vector()){
                shape = {parent.unwrap()->shape[0], parent.unwrap()->shape[0], 1, 1};
            } else if(parent.unwrap()->shape[0] != parent.unwrap()->shape[1]){
                throw InvalidArguments(name, {parent}, "Parent is not a vector or a sqyare matrix.");
            } else {
                shape = {parent.unwrap()->shape[0], 1, 1, 1};
            }
        };

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Diagonal>(graph, ancestors[0]);
        }

        Shape get_shape() const{
            return shape;
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad.diag();
        }
    };

    Node Node::diag() {
        // TODO a.daig().diag() = a
        return apply<Diagonal>(this);
    }

    Node diag(Node node){
        return node.diag();
    }

    class Reshape: public UnaryOperator{
    public:
        Shape shape;
        Reshape(GraphInPtr graph, Node parent, Shape shape):
                UnaryOperator("Reshape", graph, parent),
                shape(shape){
            SymInt product_parent = number_of_elements(parent.unwrap()->shape);
            SymInt product_shape = number_of_elements(this->shape);
            if(product_parent != product_shape){
                std::string shape_str;
                throw InvalidArguments(name, {parent.unwrap()->id}, {parent.unwrap()->shape, this->shape},
                                       "Operator 'Reshape' must not change the total number of elements");
            }
        };

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Reshape>(graph, ancestors[0], shape);
        }

        Shape get_shape() const{
            return shape;
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad.reshape(parent.unwrap()->shape);
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<Reshape> cast_op = std::static_pointer_cast<Reshape>(op);
                return symbolic_equals(parent, cast_op->parent) and shape == cast_op->shape;
            }
            return false;
        }
    };

    Node Node::reshape(Shape shape){
        std::shared_ptr<NodeInternal> ptr = unwrap();
        return ptr->graph->derived_node(std::make_shared<Reshape>(ptr->graph, this, shape));
    }

    Node reshape(Node node, Shape shape){
        return node.reshape(shape);
    }

    Node Node::flatten(size_t ndim) {
        std::shared_ptr<NodeInternal> ptr = unwrap();
        if(ndim == 0 or ndim > 4){
            throw InvalidArguments(ptr->name, ptr->op->get_parents(), "ndim = " + std::to_string(ndim)+" is outside [1,4]");
        }
        Shape shape = ptr->shape;
        for(int i=3;i>=ndim;i--){
            shape[i-1] = shape[i] * shape[i-1];
            shape[i] = 1;
        }
        return reshape(shape);
    }

    Node flatten(Node node, size_t ndim = 1){
        return node.flatten(ndim);
    }

    class Reorder: public UnaryOperator{
    public:
        std::array<size_t ,4> order;
        Reorder(GraphInPtr graph, Node parent, std::array<size_t, 4> order):
                UnaryOperator("Reorder", graph, parent),
                order(order){
            bool check[4] {false, false, false, false};
            for(int i=0;i<4;i++){
                if(order[i] > 4){
                    throw InvalidArguments(name, {this->parent},
                                           "The ordering for 'Reorder' must contain elements in the range [0,3]");
                }
                if(check[order[i]]){
                    throw InvalidArguments(name, {this->parent},
                                           "The ordering for 'Reorder' must not have repeating elements");
                }
            }
        };

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<Reorder>(graph, ancestors[0], order);
        }

        Shape get_shape() const{
            return {parent.unwrap()->shape[order[0]], parent.unwrap()->shape[order[1]],
                    parent.unwrap()->shape[order[2]], parent.unwrap()->shape[order[3]]};
        }

        static std::array<size_t ,4> reverse_order(std::array<size_t ,4>& order){
            std::array<size_t ,4> reversed;
            // 2, 0, 1, 3
            // 1, 2, 0, 3
            for(size_t i=0;i<4;i++){
                reversed[order[i]] = i;
            }
            return reversed;
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad.reorder(reverse_order(order));
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<Reorder> cast_op = std::static_pointer_cast<Reorder>(op);
                return symbolic_equals(parent, cast_op->parent) and order == cast_op->order;
            }
            return false;
        }
    };

    Node Node::reorder(std::array<size_t, 4> order){
        std::shared_ptr<NodeInternal> ptr = unwrap();
        return ptr->graph->derived_node(std::make_shared<Reorder>(ptr->graph, this, order));
    }


    Node reorder(Node node, std::array<size_t, 4> order){
        return node.reorder(order);
    }

    Node Node::reorder(size_t dim0, size_t dim1, size_t dim2, size_t dim3){
        return reorder({dim0, dim1, dim2, dim3});
    }

    Node reorder(Node node, size_t dim0, size_t dim1, size_t dim2=2, size_t dim3=3){
        return node.reorder({dim0, dim1, dim2, dim3});
    }
}
#endif //METADIFF_OPERATORS_SHAPE_H
