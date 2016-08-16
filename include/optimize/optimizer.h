#ifndef METADIFF_OPT_OPTIMIZER_H
#define METADIFF_OPT_OPTIMIZER_H

namespace metadiff{
    namespace opt {
        using namespace core;

class Optimizer {

private:
    shared_ptr<GraphInternal> _graph;

public:
    Optimizer(shared_ptr<GraphInternal> graph) : _graph(graph) {};

    void run() {
        // opt_merge();
        // opt_const_folding();
        // opt_const_elimination();
        // opt_neg_neg();
        opt_sum_scalar_martix();

        //remove all inactive nodes from graph
        _graph->removeInactiveNodes();

        // ensure the ordering of nodes in graph
        _graph->topo_sort();

        // for(auto nPtr : nodes) {
        //     std::cout<<"Node "<<nPtr->id<<std::endl;
        //     std::vector<int> parentIds{};
        //     nPtr->op->get_parents_ids(parentIds);
        //     for (auto p : parentIds)
        //         std::cout<<"parents "<<p<<std::endl;
        //     for (auto c : nPtr->children)
        //         std::cout<<"children "<<c->id<<std::endl;
        // }
    };

    void opt_merge() {
        // logger()->debug() << "1. merge...";
        // key:operator name; value:node id
        typedef unordered_map<string, Node> opMap;
        // key:parents ids
        map<vector<int>, opMap> nodeMap;

        for(Node node : _graph->nodes) {
            if(!node.is_active()) continue;

            vector<int> parentIds{};
            node->op->get_parents_ids(parentIds);
            if (parentIds.empty()) {
                //??
            }
            else {
                if (nodeMap.find(parentIds) == nodeMap.end()) {
                    nodeMap[parentIds][node->op->name] = node;
                }
                else if(nodeMap[parentIds].find(node->op->name) 
                    == nodeMap[parentIds].end())
                    nodeMap[parentIds][node->op->name] = node;
                else {
                    // associate children with existing node
                    auto existChildren = nodeMap[parentIds][node->op->name]->children;
                    existChildren.insert(existChildren.end(),
                        node->children.begin(),
                        node->children.end()); //unique????

                    // replace current node with existing node from its children's parents
                    node.replace_parent_of_children(nodeMap[parentIds][node->op->name]);

                    // remove current node from its parents's children
                    for (Node p : node->op->get_parents()){
                        p.remove_child(node);
                    }

                    // mark removal of current node
                    node.set_inactive();
                }
            }
        }        
    }

    void const_folding_dfs(Node node, unordered_set<int>& visited) {

        if (visited.find(node->id) != visited.end()) return;

        visited.insert(node->id);

        auto parents = node->op->get_parents();

        if(parents.empty()) return;

        // one parent is not constant and not deducable 
        for (Node p: parents) {
            if(!p.is_constant() and p.is_active()) {
                return;
            }
        }

        // calculate the constant value for deduction
        if (node->op->name == "Add") {
            double value = 0.0;
            for (Node p : parents) {

                if(p.is_active()) {
                    value += p->op->getConstVal();
                }

                p.remove_child(node);

                // const has no parent, if it then has no children, remove it later
                if (p->children.empty())
                    p.set_inactive();
            }

            Node newNode = node.replace_with_constant(value);
        }

        for (Node child : node->children) {
            const_folding_dfs(child, visited);
        }
    }

    void opt_const_folding() {
        // logger()->debug() << "2. opt_const_folding...";

        // ids of nodes
        unordered_set<int> visited;
        visited.reserve(_graph->nodes.size());

        for(Node node : _graph->nodes) {
            if(!node.is_active()) continue;

            const_folding_dfs(node, visited);
        }
    }

    void opt_const_elimination() {
        // logger()->debug() << "3. opt_const_elimination...";

        for(Node node : _graph->nodes) {
            if (!node.is_active()) continue;

            NodeVec parents = node->op->get_parents();

            if (parents.size()==2) { //is binary operator

                if(parents[0].is_constant() ^ parents[1].is_constant()) { //XOR

                    Node conNode(parents[0]), nConNode(parents[1]);

                    if (parents[1].is_constant())
                        std::swap(conNode, nConNode);

                    const double constVal = conNode->op->getConstVal();
                    if (constVal != 1.0 and constVal != -1.0)
                        continue;

                    if (node.is("MatrixMul") or node.is("Mul")) {
                        // mul(1,x) and mul(-1,x)
                        conNode.remove_child(node);
                        node.replace_const_eli(constVal, nConNode); //const value
                    }
                    else if (node.is("Pow") and constVal == 1.0) {
                        // pow(1,x)
                        conNode.remove_child(node);
                        node.replace_const_eli(1, nConNode); //const value
                    }

                    if (conNode->children.empty())
                        conNode.set_inactive();
                }
            }
            // To-do: for mul(1, x, y) -> mul(x, y)
        }
    }

    // div().div()?
    void opt_neg_neg() {

        for(Node node : _graph->nodes) {

            if (!node.is("Neg") or !node.is_active()) continue;

            Node parent = node->op->get_parents()[0];

            if (!parent.is("Neg") or !node.is_active()) continue;

            Node grandParent = parent->op->get_parents()[0];

            grandParent.replace_children_from(node);
            node.replace_parent_of_children(grandParent);
            node.set_inactive();
            parent.set_inactive();
        }
    }

    void opt_neg_div() {
        // the scope of this optimization is rather limited
        // do we have a div yet? 
        // current Div is Unary division (inverse)
    }

    void opt_sum_scalar_martix() {
        // sum(scalar * tensor) -> scalar * sum(tensor)

        for (Node node : _graph->nodes) {
            // consider elementwise multiplications of multiple scalars and matrices
            if (!node.is("Mul") or !node.is_active()) 
                continue;

            auto sumLoc = std::find_if(node->children.begin(), node->children.end(),
                            [](Node node) {return node.is("Sum");});

            if (sumLoc == node->children.end())
                continue;

            vector<Node> sp{}; //scalar parents
            vector<Node> mp{}; //matrix parents
            for (Node p : node->op->get_parents()) {
                if (p.is_scalar()) {
                    sp.push_back(p);
                }
                else {
                    mp.push_back(p); //not scalar then it's matrix??
                }
            }
            if (sp.empty() or mp.empty()) continue;

            if (mp.size() >1)
                node->op->update_parents(mp);
            else {
                // if the matrix is the only matrix, bypass current node
                node.set_inactive();
                mp[0].replace_children_from(node);
                node.replace_parent_of_children(mp[0]);
            }

            for (Node n: sp) {
                n.remove_child(node);
            }

            Node sumNode = *sumLoc;
            NodeVec sumNodeChildren = sumNode->children;
            sumNode->children = {};

            sp.push_back(sumNode);
            Node sMul = Node::mul(sp);
            std::cout<<"create mul node: "<<sMul->id<<std::endl;

            sMul->children = sumNodeChildren;
        }
    }

    void opt_inplace_elementwise() {

    }
          
};}}
#endif //METADIFF_OPT_OPTIMIZER_H