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
            if(not parent->is_matrix()){
                throw InvalidArguments(name, {parent}, "Parent is not a vector or a sqyare matrix.");
            }
            if(parent->is_vector()){
                shape = {parent->shape[0], parent->shape[0], 1, 1};
            } else if(parent->shape[0] != parent->shape[1]){
                throw InvalidArguments(name, {parent}, "Parent is not a vector or a sqyare matrix.");
            } else {
                shape = {parent->shape[0], 1, 1, 1};
            }
        };

        Shape get_shape(){
            return shape;
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad->diag();
        }
    };

    Node NodeInternal::diag() {
        return apply<Diagonal>();
    }

    Node diag(Node node){
        return node->diag();
    }

    class Reshape: public UnaryOperator{
    public:
        Shape shape;
        Reshape(GraphInPtr graph, Node parent, Shape shape):
                UnaryOperator("Reshape", graph, parent),
                shape(shape){
            auto product_parent = number_of_elements(parent->shape);
            auto product_shape = number_of_elements(this->shape);
            if(product_parent != product_shape){
                std::string shape_str;
                throw InvalidArguments(name, {parent->id}, {parent->shape, this->shape},
                                       "Operator 'Reshape' must not change the total number of elemetns");
            }
        };

        Shape get_shape(){
            return shape;
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad->reshape(parent->shape);
        }
    };

    Node NodeInternal::reshape(Shape shape){
        return graph->derived_node(std::make_shared<Reshape>(graph, this, shape));
    }

    Node reshape(Node node, Shape shape){
        return node->reshape(shape);
    }

    Node NodeInternal::flatten(size_t ndim) {
        if(ndim == 0 or ndim > 4){
            throw InvalidArguments(name, op->get_parents(), "ndim = " + std::to_string(ndim)+" is outside [1,4]");
        }
        Shape shape = this->shape;
        for(int i=3;i>=ndim;i--){
            shape[i-1] = shape[i] * shape[i-1];
            shape[i] = 1;
        }
        return reshape(shape);
    }

    Node flatten(Node node, size_t ndim = 1){
        return node->flatten(ndim);
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

        Shape get_shape(){
            return {parent->shape[order[0]], parent->shape[order[1]],
                    parent->shape[order[2]], parent->shape[order[3]]};
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
            return my_grad->reorder(reverse_order(order));
        }
    };

    Node NodeInternal::reorder(std::array<size_t, 4> order){
        graph->derived_node(std::make_shared<Reorder>(graph, this, order));
    }


    Node reorder(Node node, std::array<size_t, 4> order){
        return node->reorder(order);
    }

    Node NodeInternal::reorder(size_t dim0, size_t dim1, size_t dim2, size_t dim3){
        return reorder({dim0, dim1, dim2, dim3});
    }

    Node reorder(Node node, size_t dim0, size_t dim1, size_t dim2=2, size_t dim3=3){
        return node->reorder({dim0, dim1, dim2, dim3});
    }
}
#endif //METADIFF_OPERATORS_SHAPE_H
