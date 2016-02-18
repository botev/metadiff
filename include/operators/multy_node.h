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
                UnaryOperator(name, graph, parent) {}

        ad_value_type get_value_type() const{
            return results_v_types[0];
        }

        Shape get_shape() const{
            return results_shapes[0];
        }

        ad_node_type get_node_type() const{
            return results_types[0];
        };
    };

    /**
     * An operator which selects one of the children of a MultiNode operator
     */
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
            std::shared_ptr<MultiNode> multi_op = std::dynamic_pointer_cast<MultiNode>(parent.unwrap()->op);
            if(not multi_op){
                throw InvalidArguments(name, {parent}, "Parent must be a result of an operator of type 'MultiNode'.");
            }
            if(index >= multi_op->results_shapes.size()){
                throw InvalidArguments(name, {parent}, "Provided index is too big: " + std::to_string(index));
            }
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<MultiNodeIndex>(graph, ancestors[0], index);
        }

        ad_value_type get_value_type() const{
            std::shared_ptr<MultiNode> multi_op = std::static_pointer_cast<MultiNode>(parent.unwrap()->op);
            return multi_op->results_v_types[index];
        }

        Shape get_shape() const{
            std::shared_ptr<MultiNode> multi_op = std::static_pointer_cast<MultiNode>(parent.unwrap()->op);
            return multi_op->results_shapes[index];
        }

        ad_node_type get_node_type() const{
            std::shared_ptr<MultiNode> multi_op = std::static_pointer_cast<MultiNode>(parent.unwrap()->op);
            return multi_op->results_types[index];
        };

        size_t get_gradient_level() const{
            return parent.unwrap()->grad_level;
        }

        NodeVec get_parents() const{
            return {parent};
        }

        NodeVec get_arguments() const{
            return NodeVec {};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            return my_grad;
        }

        bool equals(const std::shared_ptr<Operator> op) const{
            if(name == op->name){
                std::shared_ptr<MultiNodeIndex> cast_op = std::static_pointer_cast<MultiNodeIndex>(op);
                return symbolic_equals(parent, cast_op->parent) and index == cast_op->index;
            }
            return false;
        }
    };

    /**
     * Max and argmax operator
     */
    class MaxAndArgMax: public MultiNode {
    public:
        size_t axis;
        MaxAndArgMax(GraphInPtr graph,
                     Node parent, size_t axis):
                MultiNode("MaxAndArgMax", graph, parent),
                axis(axis){
            if(parent.unwrap()->v_type == BOOLEAN){
                throw InvalidArguments(name, {parent}, "Parent can not be of type BOOLEAN");
            }
            if(parent.unwrap()->type == SYMBOLIC_INTEGER){
                throw InvalidArguments(name, {parent}, "Parent can not be of type SYMBOLIC_INTEGER");
            }
            Shape shape = parent.unwrap()->shape;
            shape[axis] = 1;
            this->results_shapes = {shape, shape};
            if(parent.unwrap()->type == INPUT or parent.unwrap()->type == SHARED_INPUT or parent.unwrap()->type == INPUT_DERIVED){
                this->results_types = {INPUT_DERIVED, CONSTANT_DERIVED};
            } else if(parent.unwrap()->type == CONSTANT_DERIVED){
                this->results_types = {CONSTANT_DERIVED, CONSTANT_DERIVED};
            } else {
                this->results_types = {CONSTANT, CONSTANT};
            }
            this->results_v_types = {parent.unwrap()->v_type, INTEGER};
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<MaxAndArgMax>(graph, ancestors[0], axis);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            // Here the my_grad is the grad with respect to the max
            return graph->derived_node(std::make_shared<IndexGrad>(graph, my_grad, owner.argmax(axis), axis, owner.unwrap()->shape[axis]));
        }
    };

    Node Node::max(size_t axis) {
        if(axis == AUTO_INFER_AXIS){
            for(size_t i = 0; i < 4; i++){
                if(unwrap()->shape[3-i] != 1){
                    axis = 3 - i;
                    break;
                }
            }
        }
        Node max_and_arg_max = unwrap()->graph->derived_node(
                std::make_shared<MaxAndArgMax>(unwrap()->graph, this, axis));
        return unwrap()->graph->derived_node(std::make_shared<MultiNodeIndex>(unwrap()->graph, max_and_arg_max, 0));
    }

    Node Node::argmax(size_t axis) {
        if(axis == AUTO_INFER_AXIS){
            for(size_t i = 0; i < 4; i++){
                if(unwrap()->shape[3-i] != 1){
                    axis = 3 - i;
                    break;
                }
            }
        }
        Node max_and_arg_max = unwrap()->graph->derived_node(
                std::make_shared<MaxAndArgMax>(unwrap()->graph, this, axis));
        return unwrap()->graph->derived_node(std::make_shared<MultiNodeIndex>(unwrap()->graph, max_and_arg_max, 1));
    }

    /**
     * Sort and argsort operator
     */
    class SortAndArgSort: public MultiNode {
    public:
        size_t axis;
        SortAndArgSort(GraphInPtr graph,
                       Node parent, size_t axis):
                MultiNode("SortAndArgSort", graph, parent),
                axis(axis){
            if(parent.unwrap()->v_type == BOOLEAN){
                throw InvalidArguments(name, {parent}, "Parent can not be of type BOOLEAN");
            }
            if(parent.unwrap()->type == SYMBOLIC_INTEGER){
                throw InvalidArguments(name, {parent}, "Parent can not be of type SYMBOLIC_INTEGER");
            }
            Shape shape = parent.unwrap()->shape;
            shape[axis] = 1;
            this->results_shapes = {shape, shape};
            if(parent.unwrap()->type == INPUT or parent.unwrap()->type == SHARED_INPUT or parent.unwrap()->type == INPUT_DERIVED){
                this->results_types = {INPUT_DERIVED, CONSTANT_DERIVED};
            } else if(parent.unwrap()->type == CONSTANT_DERIVED){
                this->results_types = {CONSTANT_DERIVED, CONSTANT_DERIVED};
            } else {
                this->results_types = {CONSTANT, CONSTANT};
            }
            this->results_v_types = {parent.unwrap()->v_type, INTEGER};
        }

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors) const{
            return std::make_shared<MaxAndArgMax>(graph, ancestors[0], axis);
        }

        Node get_parent_grad(Node my_grad, size_t index){
            // Here the my_grad is the grad with respect to the max
            return graph->derived_node(std::make_shared<IndexGrad>(graph, my_grad, owner.argsort(axis), axis, owner.unwrap()->shape[axis]));
        }
    };

    Node Node::sort(size_t axis) {
        if(axis == AUTO_INFER_AXIS){
            for(size_t i = 0; i < 4; i++){
                if(unwrap()->shape[3-i] != 1){
                    axis = 3 - i;
                    break;
                }
            }
        }
        Node max_and_arg_max = unwrap()->graph->derived_node(
                std::make_shared<SortAndArgSort>(unwrap()->graph, this, axis));
        return unwrap()->graph->derived_node(std::make_shared<MultiNodeIndex>(unwrap()->graph, max_and_arg_max, 0));
    }

    Node Node::argsort(size_t axis){
        if(axis == AUTO_INFER_AXIS){
            for(size_t i = 0; i < 4; i++){
                if(unwrap()->shape[3-i] != 1){
                    axis = 3 - i;
                    break;
                }
            }
        }
        Node max_and_arg_max = unwrap()->graph->derived_node(
                std::make_shared<SortAndArgSort>(unwrap()->graph, this, axis));
        return unwrap()->graph->derived_node(std::make_shared<MultiNodeIndex>(unwrap()->graph, max_and_arg_max, 1));
    }
}
#endif //METADIFF_OPERATORS_MULTY_NODE_H
