//
// Created by alex on 19/12/15.
//

#ifndef METADIFF_OPERATORS_OPTIMIZED_H
#define METADIFF_OPERATORS_OPTIMIZED_H

namespace metadiff {

    class BinaryCrossEntropyLogit : public ElementwiseNary {
    public:
        BinaryCrossEntropyLogit(GraphInPtr graph, NodeVec parents):
                ElementwiseNary("BinCrossEntropyLogit", graph, parents)
        {
            if(parents.size() != 4){
                throw InvalidArguments(name, parents, "Operator 'BinCrossEntropyLogit' takes exactly 4 arguments.");
            }
        }

        ad_value_type get_value_type() {
            return FLOAT;
        };

        Node get_parent_grad(Node my_grad, size_t index){
            // TODO
            return my_grad;
        }

        void generate_gradients(std::vector<Node>& messages) {
            // Check for any incoming messages
            if(messages[owner.ptr->id].empty()){
                return;
            }

            // Get the gradient with respect to this node
            Node my_grad = messages[owner.ptr->id];
            // Update the message name
            if(my_grad.ptr->name == "Derived Node" or my_grad.ptr->name == ""){
                my_grad.ptr->name = "Grad of " + std::to_string(owner.ptr->id);
            } else {
                my_grad.ptr->name += "|Grad of " + std::to_string(owner.ptr->id);
            }

            // Check for any surprises, where all parents are constants
            // If that is the case this node should have been constant as well
            // and no message should have been sent to it
            NodeVec parents = get_parents();
            bool constant = true;
            for(int i=0;i<parents.size();i++){
                if(not parents[i].is_constant()){
                    constant = false;
                    break;
                }
            }
            if(constant){
                throw UnknownError({parents}, "Gradient message present, all parents are constants");
            }

            // Parents - 1)p, 2) sf(x), 3) sf(-x), 4)x
            // Node computes f = - p * log(q) - (1-p) * log(1-q)
            // log(q) = -sf(-x), log(1-q) = -sf(x)
            // Node computes f = p * sf(-x) + (1 - p)*sf(x) = p*(sf(-x)-sf(x)) + sf(x)
            // dE/dp = dE * (sf(-x)-sf(x))
            // dE/dx = dE * (q-p) = dE * (sigmoid(x) - p)
            Node p = parents[0];
            Node sfx = parents[1];
            Node sfmx = parents[2];
            Node x = parents[3];
            if(not p.is_constant()){
                Node msfx = sfx.neg();
                msfx.update_grad_level();
                Node parent_grad = mul(my_grad, add(sfmx, msfx));
                parent_grad.ptr->name =
                        "Grad msg " + std::to_string(owner.ptr->id) + " -> " + std::to_string(p.ptr->id);
                send_grad_message(p.ptr->id, parent_grad, messages);
            }
            if(not x.is_constant()){
                Node sigmoid = x.sigmoid();
                sigmoid.update_grad_level();
                Node neg = p.neg();
                neg.update_grad_level();
                Node parent_grad = mul(my_grad, add(sigmoid, neg));
                parent_grad.ptr->name =
                        "Grad msg " + std::to_string(owner.ptr->id) + " -> " + std::to_string(x.ptr->id);
                send_grad_message(x.ptr->id, parent_grad, messages);
            }
        }
    };

    Node binary_cross_entropy_logit(Node p, Node x){
        return apply<BinaryCrossEntropyLogit>(NodeVec{p, softplus(p), softplus(p.neg()), x});
    }

    Node Node::relu(){
        Node half = ptr->graph->constant_value(0.5);
        return mul(half, add(this, abs()));
    }

    Node relu(Node x){
        return x.relu();
    }
};


#endif //METADIFF_OPERATORS_OPTIMIZED_H
