//
// Created by alex on 15/12/15.
//

#ifndef AUTODIFF_MULTY_NODE_H
#define AUTODIFF_MULTY_NODE_H
namespace metadiff{
    // A common super class for special operators with more than 1 output
    // such as MaxAndArgMax and SortAndArgSort
    // Because of how the gradients are set up only one node can be differentiable
    class MultiNode : public UnaryOperator{
    public:
        NodeInPtr parent;
        std::vector<Shape> results_shapes;
        std::vector<ad_node_type> results_types;
        std::vector<ad_value_type> results_v_types;
        MultiNode(std::string const name,
                  GraphInPtr graph,
                  NodeInPtr parent):
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
        NodeInPtr parent;
        size_t index;
        MultiNodeIndex(GraphInPtr graph,
                       NodeInPtr parent,
                       size_t index):
                Operator("MultyNodeIndex", graph),
                parent(parent),
                index(index)
        {
            auto parent_op = parent.lock()->op;
            MultiNode* multi_op = dynamic_cast<MultiNode*>(parent_op.get());
            if(not multi_op){
                throw UnknownError({parent}, "The operator 'MultiNodeIndex' can be applied only to nodes, "
                        "whose operators are subclasses of 'MultiNode'");
            }
            if(index >= multi_op->results_shapes.size()){
                throw InvalidArguments(name, {parent}, "Provided index is too big: " + std::to_string(index));
            }
        }

        ad_value_type get_value_type(){
            auto parent_op = parent.lock()->op;
            MultiNode* multi_op = dynamic_cast<MultiNode*>(parent_op.get());
            return multi_op->results_v_types[index];
        }

        Shape get_shape(){
            auto parent_op = parent.lock()->op;
            MultiNode* multi_op = dynamic_cast<MultiNode*>(parent_op.get());
            return multi_op->results_shapes[index];
        }

        ad_node_type get_node_type(){
            auto parent_op = parent.lock()->op;
            MultiNode* multi_op = dynamic_cast<MultiNode*>(parent_op.get());
            return multi_op->results_types[index];
        };

        unsigned short get_gradient_level(){
            return parent.lock()->grad_level;
        }

        NodeInVec get_parents(){
            return {parent};
        }

        NodeInVec get_arguments(){
            return NodeInVec {};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t, size_t>& messages){
            auto graph = this->graph.lock();

            // Check for any incoming messages
            if(messages.find(current) == messages.end()){
                return;
            }

            // Get the gradient with respect to this node
            auto my_grad = graph->nodes[messages[current]];
            update_grad_name(my_grad, current);

            auto parent = this->parent.lock();
            if(parent->is_constant()){
                throw UnknownError({parent}, "Gradient message present, but parents are " + to_string(parent->type));
            }

            auto parent_grad = my_grad;
            send_grad_message(graph, parent->id, parent_grad->id, messages);
        }
    };

    // First node is Max and second is ArgMax
    class MaxAndArgMax: public MultiNode {
    public:
        std::vector<size_t> axes;
        MaxAndArgMax(GraphInPtr graph,
                     NodeInPtr parent, std::vector<size_t> axes):
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
            auto parent_node = this->parent.lock();
            if(parent_node->v_type == BOOLEAN){
                throw InvalidArguments(name, {parent}, "Operator 'MaxAndArgMax' can not be "
                "applied to a BOOLEAN node");
            }
            if(parent_node->type == SYMBOLIC_INTEGER){
                throw InvalidArguments(name, {parent}, "Operator 'MaxAndArgMax' can not be "
                        "applied to a SYMBOLIC_INTEGER node");
            }
            Shape shape = parent_node->shape;
            for(int i=0;i<axes.size();i++){
                shape[axes[i]] = 1;
            }
            this->results_shapes = {shape, shape};
            if(parent_node->type == INPUT or parent_node->type == SHARED_INPUT or parent_node->type == INPUT_DERIVED){
                this->results_types = {INPUT_DERIVED, CONSTANT_DERIVED};
            } else if(parent_node->type == CONSTANT_DERIVED){
                this->results_types = {CONSTANT_DERIVED, CONSTANT_DERIVED};
            } else {
                this->results_types = {CONSTANT, CONSTANT};
            }
            this->results_v_types = {parent_node->v_type, INTEGER};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t , size_t>& messages){
            // TODO
        }
    };

    // First node is Max and second is ArgMax
    class SortAndArgSort: public MultiNode {
    public:
        std::vector<size_t> axes;
        SortAndArgSort(GraphInPtr graph,
                     NodeInPtr parent, std::vector<size_t> axes):
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
            auto parent_node = this->parent.lock();
            if(parent_node->v_type == BOOLEAN){
                throw InvalidArguments(name, {parent}, "Operator 'SortAndArgSort' can not be "
                        "applied to a BOOLEAN node");
            }
            if(parent_node->type == SYMBOLIC_INTEGER){
                throw InvalidArguments(name, {parent}, "Operator 'SortAndArgSort' can not be "
                        "applied to a SYMBOLIC_INTEGER node");
            }
            Shape shape = parent_node->shape;
            this->results_shapes = {shape, shape};
            if(parent_node->type == INPUT or parent_node->type == SHARED_INPUT or parent_node->type == INPUT_DERIVED){
                this->results_types = {INPUT_DERIVED, CONSTANT_DERIVED};
            } else if(parent_node->type == CONSTANT_DERIVED){
                this->results_types = {CONSTANT_DERIVED, CONSTANT_DERIVED};
            } else {
                this->results_types = {CONSTANT, CONSTANT};
            }
            this->results_v_types = {parent_node->v_type, INTEGER};
        }

        void generate_gradients(size_t current, std::unordered_map<size_t , size_t>& messages){
            // TODO
        }
    };
}
#endif //AUTODIFF_MULTY_NODE_H
