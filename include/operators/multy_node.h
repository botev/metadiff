//
// Created by alex on 15/12/15.
//

#ifndef METADIFF_OPERATORS_MULTY_NODE_H
#define METADIFF_OPERATORS_MULTY_NODE_H

namespace metadiff{
    // A common super class for special operators with more than 1 output
    // such as MaxAndArgMax and SortAndArgSort
    // Because of how the gradients are set up only one node can be differentiable
    class MultiNode : public UnaryOperator{
    public:
        Node parent;
        std::vector<Shape> results_shapes;
        std::vector<ad_node_type> results_types;
        std::vector<ad_value_type> results_v_types;
        MultiNode(std::string const name,
                  GraphInPtr graph,
                  Node parent):
                UnaryOperator(name, graph, parent)
        {}

        ad_value_type get_value_type(){
            return results_v_types[0];
        }

        Shape get_shape(){
            return results_shapes[0];
        }

        ad_node_type get_node_type(){
            return results_types[0];
        };
    };


// Special operator for nodes which consist of more than 1 value
// Such as MaxAndArgMax, SortAndArgSort
    class MultiNodeIndex : public Operator {
    public:
        Node parent;
        size_t index;
        MultiNodeIndex(GraphInPtr graph,
                       Node parent,
                       size_t index):
                Operator("MultyNodeIndex", graph),
                parent(parent),
                index(index)
        {
            MultiNode* multi_op = dynamic_cast<MultiNode*>(parent.ptr->op.get());
            if(not multi_op){
                throw UnknownError({parent}, "The operator 'MultiNodeIndex' can be applied only to nodes, "
                        "whose operators are subclasses of 'MultiNode'");
            }
            if(index >= multi_op->results_shapes.size()){
                throw InvalidArguments(name, {parent}, "Provided index is too big: " + std::to_string(index));
            }
        }

        ad_value_type get_value_type(){
            MultiNode* multi_op = dynamic_cast<MultiNode*>(parent.ptr->op.get());
            return multi_op->results_v_types[index];
        }

        Shape get_shape(){
            MultiNode* multi_op = dynamic_cast<MultiNode*>(parent.ptr->op.get());
            return multi_op->results_shapes[index];
        }

        ad_node_type get_node_type(){
            MultiNode* multi_op = dynamic_cast<MultiNode*>(parent.ptr->op.get());
            return multi_op->results_types[index];
        };

        size_t get_gradient_level(){
            return parent.ptr->grad_level;
        }

        NodeVec get_parents(){
            return {parent};
        }

        NodeVec get_arguments(){
            return NodeVec {};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }
    };

    // First node is Max and second is ArgMax
    class MaxAndArgMax: public MultiNode {
    public:
        std::vector<size_t> axes;
        MaxAndArgMax(GraphInPtr graph,
                     Node parent, std::vector<size_t> axes):
                MultiNode("MaxAndArgMax", graph, parent),
                axes(axes){
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
            if(parent.ptr->v_type == BOOLEAN){
                throw InvalidArguments(name, {parent}, "Operator 'MaxAndArgMax' can not be "
                        "applied to a BOOLEAN node");
            }
            if(parent.ptr->type == SYMBOLIC_INTEGER){
                throw InvalidArguments(name, {parent}, "Operator 'MaxAndArgMax' can not be "
                        "applied to a SYMBOLIC_INTEGER node");
            }
            Shape shape = parent.ptr->shape;
            for(int i=0;i<axes.size();i++){
                shape[axes[i]] = 1;
            }
            this->results_shapes = {shape, shape};
            if(parent.ptr->type == INPUT or parent.ptr->type == SHARED_INPUT or parent.ptr->type == INPUT_DERIVED){
                this->results_types = {INPUT_DERIVED, CONSTANT_DERIVED};
            } else if(parent.ptr->type == CONSTANT_DERIVED){
                this->results_types = {CONSTANT_DERIVED, CONSTANT_DERIVED};
            } else {
                this->results_types = {CONSTANT, CONSTANT};
            }
            this->results_v_types = {parent.ptr->v_type, INTEGER};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            // TODO
            return my_grad;
        }
    };

    // First node is Max and second is ArgMax
    class SortAndArgSort: public MultiNode {
    public:
        std::vector<size_t> axes;
        SortAndArgSort(GraphInPtr graph,
                       Node parent, std::vector<size_t> axes):
                MultiNode("SortAndArgSort", graph, parent),
                axes(axes){
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
            if(parent.ptr->v_type == BOOLEAN){
                throw InvalidArguments(name, {parent}, "Operator 'SortAndArgSort' can not be "
                        "applied to a BOOLEAN node");
            }
            if(parent.ptr->type == SYMBOLIC_INTEGER){
                throw InvalidArguments(name, {parent}, "Operator 'SortAndArgSort' can not be "
                        "applied to a SYMBOLIC_INTEGER node");
            }
            Shape shape = parent.ptr->shape;
            this->results_shapes = {shape, shape};
            if(parent.ptr->type == INPUT or parent.ptr->type == SHARED_INPUT or parent.ptr->type == INPUT_DERIVED){
                this->results_types = {INPUT_DERIVED, CONSTANT_DERIVED};
            } else if(parent.ptr->type == CONSTANT_DERIVED){
                this->results_types = {CONSTANT_DERIVED, CONSTANT_DERIVED};
            } else {
                this->results_types = {CONSTANT, CONSTANT};
            }
            this->results_v_types = {parent.ptr->v_type, INTEGER};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            // TODO
            return my_grad;
        }
    };
}
#endif //METADIFF_OPERATORS_MULTY_NODE_H
