//
// Created by alex on 19/12/15.
//

#ifndef METADIFF_OPERATORS_OPTIMIZED_H
#define METADIFF_OPERATORS_OPTIMIZED_H

namespace metadiff {

    class BinaryCrossEntropyLogit : public ElementwiseBinary {
    public:
        Node softplus_x, softplus_mx;
        BinaryCrossEntropyLogit(GraphInPtr graph, Node p, Node x):
                ElementwiseBinary("BinCrossEntropyLogit", graph, p, x)
        {
            softplus_x = softplus(x);
            softplus_mx = softplus(-x);
        }

        BinaryCrossEntropyLogit(GraphInPtr graph, Node p, Node x,
                Node softplus_x, Node softplus_mx):
                ElementwiseBinary("BinCrossEntropyLogit", graph, p, x),
                softplus_x(softplus_x),
                softplus_mx(softplus_mx) {};

        std::shared_ptr<Operator> copy_to(GraphInPtr graph, std::vector<Node> ancestors){
            return std::make_shared<BinaryCrossEntropyLogit>(graph,
                                                             ancestors[0], ancestors[1],
                                                             ancestors[2], ancestors[3]);
        }

        ad_value_type get_value_type() {
            return FLOAT;
        };

        NodeVec get_arguments(){
            return {softplus_x, softplus_mx};
        }

        Node get_parent_grad(Node my_grad, size_t index){
            // Parents - p, x
            // Arguments - sf(x), sf(-x)
            // Node computes f = - p * log(q) - (1-p) * log(1-q)
            // log(q) = -sf(-x), log(1-q) = -sf(x)
            // Node computes f = p * sf(-x) + (1 - p)*sf(x) = p*(sf(-x)-sf(x)) + sf(x)
            // dE/dp = dE * (sf(-x)-sf(x))
            // dE/dx = dE * (q-p) = dE * (sigmoid(x) - p)
            if(index == 0){
                return my_grad * (softplus_mx - softplus_x);
            } else {
                return my_grad * (sigmoid(parent2) - parent1);
            }
        }
    };

    Node binary_cross_entropy_logit(Node p, Node x){
        return apply<BinaryCrossEntropyLogit>(p, x);
    }

    Node Node::relu(){
        std::shared_ptr<NodeInternal> ptr = unwrap();
        return ptr->graph->constant_value(0.5) * (this + abs());
    }

    Node relu(Node x){
        return x.relu();
    }
};


#endif //METADIFF_OPERATORS_OPTIMIZED_H
