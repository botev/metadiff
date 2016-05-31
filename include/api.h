//
// Created by alex on 03/05/16.
//

#ifndef METADIFF_API_H
#define METADIFF_API_H
namespace metadiff{
    namespace api{
        using shared::SharedPtr;
        using core::GROUP_ROOT;
        using core::GROUP_DELIMITER;
        using core::AUTO_INFER_AXIS;
        using core::Axes;
        using core::SymInt;
        using core::Shape;
        using core::nodeType;
        using core::dType ;
        using core::errorPolicy;
        using core::deviceType;
        using core::Device;
        using core::Group;
        using core::Node;
        using core::NodeVec;
        using core::Update;
        using core::Updates;
        using core::Graph;

        using logging::metadiff_sink;
        using dagre::dagre_to_file;
#ifdef AFAPI
        typedef backend::ArrayfireBackend AfBackend;
#endif
        // API functions
        /** Creates a new symbolic Graph instance */
        Graph create_graph(){
            return std::make_shared<core::GraphInternal>();
        }




        // Replication of many function as sometimes it is more intuitive func(obj) then obj.func()

        // Constant operators

        Node as_constant(Node node){
            return node.as_constant();
        }

        // Base operators

        Node alias(Node node) {
            return node.alias();
        }

//        Node add(Node node1, Node node2) {
//            return Node::add(NodeVec{node1, node2});
//        };
//
//        Node neg(Node node1, Node node2) {
//            return Node::add(NodeVec{node1, node2.neg()});
//        }
//
//        Node mul(Node node1, Node node2) {
//            return Node::mul(NodeVec{node1, node2});
//        }
//
//        Node div(Node node1, Node node2) {
//            return Node::mul(NodeVec{node1, node2.div()});
//        }

        // Logical operators

        Node logical_not(Node node) {
            return node.logical_not();
        }

        Node logical_and(Node node1, Node node2) {
            return node1.logical_and(node2);
        }

        Node logical_or(Node node1, Node node2) {
            return node1.logical_or(node2);
        }

//        Node gt(Node node1, Node node2) {
//            return node1.gt(node2);
//        }
//
//        Node ge(Node node1, Node node2) {
//            return node1.ge(node2);
//        }
//
//        Node lt(Node node1, Node node2) {
//            return node1.lt(node2);
//        }
//
//        Node le(Node node1, Node node2) {
//            return node1.le(node2);
//        }
//
//        Node eq(Node node1, Node node2) {
//            return node1.eq(node2);
//        }
//
//        Node neq(Node node1, Node node2) {
//            return node1.neq(node2);
//        }

        Node approx_eq(Node node1, Node node2, double tol = 0.00001) {
            return node1.approx_eq(node2, tol);
        }

        template <typename L, typename = std::enable_if<not std::is_same<L, Node>::value>>
        Node approx_eq(L node1, Node node2, double tol = 0.00001) {
            return node2->graph->wrap(node1).approx_eq(node2, tol);
        };

        template <typename R, typename = std::enable_if<not std::is_same<R, Node>::value>>
        Node approx_eq(Node node1, R node2, double tol = 0.00001) {
            return node1.approx_eq(node1->graph->wrap(node2), tol);
        };

        Node approx_neq(Node node1, Node node2, double tol = 0.00001) {
            return node1.approx_neq(node2, tol);
        }

        template <typename L, typename = std::enable_if<not std::is_same<L, Node>::value>>
        Node approx_neq(L node1, Node node2, double tol = 0.00001) {
            return node2->graph->wrap(node1).approx_neq(node2, tol);
        };

        template <typename R, typename = std::enable_if<not std::is_same<R, Node>::value>>
        Node approx_neq(Node node1, R node2, double tol = 0.00001) {
            return node1.approx_neq(node1->graph->wrap(node2), tol);
        };

        Node is_nan(Node node) {
            return node.is_nan();
        }

        Node is_inf(Node node) {
            return node.is_inf();
        }

        Node all(Node node) {
            return node.all();
        }

        Node any(Node node) {
            return node.any();
        }

        Node select(Node condition, Node result_true, Node result_false) {
            return condition.select(result_true, result_false);
        }

        // Elementwise operators

        Node square(Node node) {
            return node.square();
        }

        Node exp(Node node) {
            return node.exp();
        }

        Node sigmoid(Node node) {
            return node.sigmoid();
        }

        Node log(Node node) {
            return node.log();
        }

        Node log10(Node node) {
            return node.log10();
        }

        Node abs(Node node) {
            return node.abs();
        }

        Node log1p(Node node) {
            return node.log1p();
        }

        Node softplus(Node node, int threshold = 50) {
            return node.softplus(threshold);
        }

        Node sin(Node node) {
            return node.sin();
        }

        Node cos(Node node) {
            return node.cos();
        }

        Node tan(Node node) {
            return node.tan();
        }

        Node cot(Node node) {
            return node.cot();
        }

        Node sinh(Node node) {
            return node.sinh();
        }

        Node cosh(Node node) {
            return node.cosh();
        }

        Node tanh(Node node) {
            return node.tanh();
        }

        Node coth(Node node) {
            return node.coth();
        }

        Node pow(Node node, Node power) {
            return node.pow(power);
        }

        template <typename L, typename = std::enable_if<not std::is_same<L, Node>::value>>
        Node pow(L node, Node power) {
            return power->graph->wrap(node).pow(power);
        };

        template <typename R, typename = std::enable_if<not std::is_same<R, Node>::value>>
        Node pow(Node node, R power) {
            return node.pow(node->graph->wrap(power));
        };


        // Shape operators

        Node diag(Node node) {
            return node.diag();
        }

        Node reshape(Node node, Shape shape) {
            return node.reshape(shape);
        }

        Node flatten(Node node, unsigned short dims = 1) {
            return node.flatten(dims);
        }

        Node reorder(Node node, Axes order) {
            return node.reorder(order);
        }

        Node reorder(Node node, short dim0, short dim1, short dim2 = 2, short dim3 = 3) {
            return node.reorder(Axes{dim0, dim1, dim2, dim3});
        }

        // Linear algebra operators

        Node transpose(Node node) {
            return node.transpose();
        }

        Node dot(NodeVec nodes) {
            return Node::dot(nodes);
        }

        Node dot(Node node1, Node node2) {
            return dot(NodeVec{node1, node2});
        }

#ifdef AFAPI
        Node dot(af::array node1, Node node2) {
            return dot(NodeVec{node2->graph->constant_value(node1), node2});
        }

        Node dot(Node node1, af::array node2) {
            return dot(NodeVec{node1, node1->graph->constant_value(node2)});
        }
#endif

        Node minv(Node node) {
            return node.minv();
        }

        Node det(Node node) {
            return node.det();
        }

        Node logdet(Node node) {
            return node.logdet();
        }

        Node trace(Node node) {
            return node.logdet();
        }

        // Index operators

        // Multi-node operators

        Node max(Node node, short axis = AUTO_INFER_AXIS){
            return node.max(axis);
        }

        Node argMax(Node node, short axis = AUTO_INFER_AXIS){
            return node.argMax(axis);
        }

        std::pair<Node,Node> maxAndArgMax(Node node, short axis = AUTO_INFER_AXIS){
            return node.maxAndArgMax(axis);
        };

        Node sort(Node node, short axis = AUTO_INFER_AXIS){
            return node.sort(axis);
        }

        Node argSortMax(Node node, short axis = AUTO_INFER_AXIS){
            return node.argSort(axis);
        }

        std::pair<Node,Node> sortAndArgSort(Node node, short axis = AUTO_INFER_AXIS){
            return node.sortAndArgSort(axis);
        };

        // Optimized operators
        Node binary_cross_entropy_logit(Node p, Node x) {
            return p.binary_cross_entropy_logit(x);
        }

        template <typename L, typename = std::enable_if<not std::is_same<L, Node>::value>>
        Node binary_cross_entropy_logit(L p, Node x) {
            return x->graph->wrap(p).binary_cross_entropy_logit(x);
        };

        template <typename R, typename = std::enable_if<not std::is_same<R, Node>::value>>
        Node binary_cross_entropy_logit(Node p, R x) {
            return p.binary_cross_entropy_logit(p->graph->wrap(x));
        };

        Node relu(Node node) {
            return node.relu();
        }
    }
}
#endif //METADIFF_API_H
