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
                // To-Do: gonna ensure the Ids are consercutive
            };

            void opt_merge() {

                // logger()->debug() << "1. merge...";
                // key:operator name; value:node id
                typedef unordered_map<string, Node> opMap;
                // key:parents ids
                map<vector<int>, opMap> nodeMap;

                for(auto nPtr : _graph->nodes) {//from member

                    vector<int> parentIds{};
                    nPtr->op->get_parents_ids(parentIds);
                    if (parentIds.empty()) {
                        //??
                    }
                    else {
                        if (nodeMap.find(parentIds) == nodeMap.end()) {
                            nodeMap[parentIds][nPtr->op->name] = nPtr;
                        }
                        else if(nodeMap[parentIds].find(nPtr->op->name) 
                            == nodeMap[parentIds].end())
                            nodeMap[parentIds][nPtr->op->name] = nPtr;
                        else {
                            // associate children with existing node
                            auto existChildren = nodeMap[parentIds][nPtr->op->name]->children;
                            existChildren.insert(existChildren.end(),
                                nPtr->children.begin(),
                                nPtr->children.end()); //unique????

                            // replace current node with existing node from its children's parents
                            Node current = nPtr;
                            current.replace_parent_from_children(nodeMap[parentIds][nPtr->op->name]);

                            // remove current node from its parents's children
                            for (auto p : nPtr->op->get_parents()){
                                p.remove_child(nPtr);
                            }

                            // mark removal of current node
                            nPtr->active = false;
                        }
                    }
                }

                //remove all inactive nodes from graph
                _graph->removeInactiveNodes();
            }

            void const_folding_dfs(Node node, unordered_set<int>& visited) {

                if (visited.find(node->id) != visited.end()) return;

                visited.insert(node->id);

                auto parents = node->op->get_parents();

                if(parents.empty()) return;

                // one parent is not constant and not deducable 
                for (auto p: parents) {
                    if(!p.is_constant() and p->active) {
                        return;
                    }
                }

                // calculate the constant value for deduction
                if (node->op->name == "Add") {
                    double value = 0.0;
                    for (auto p : parents) {

                        if(p->active) {
                            value += p->op->getConstVal();
                        }

                        p.remove_child(node);

                        // const has no parent, if it then has no children, remove it later
                        if (p->children.empty())
                            p->active = false;
                    }

                    Node newNode = node.replace_with_constant(value);
                }

                for (auto child : node->children) {
                    const_folding_dfs(child, visited);
                }
            }

            void opt_const_folding() {
                // logger()->debug() << "2. opt_const_folding...";

                // ids of nodes
                unordered_set<int> visited;
                visited.reserve(_graph->nodes.size());

                for(auto nPtr : _graph->nodes) {
                    const_folding_dfs(nPtr, visited);
                }

                _graph->removeInactiveNodes();
            }

            void opt_const_elimination() {
                // logger()->debug() << "3. opt_const_elimination...";

                for(auto nPtr : _graph->nodes) {

                    Node node = nPtr;
                    NodeVec parents = nPtr->op->get_parents();

                    if (parents.size()==2) { //is binary operator

                        if(parents[0].is_constant() ^ parents[1].is_constant()) { //XOR

                            Node conNode(parents[0]), nConNode(parents[1]);

                            if (parents[1].is_constant())
                                std::swap(conNode, nConNode);

                            const double constVal = conNode->op->getConstVal();
                            const string opName = node->op->name;

                            if (constVal != 1.0 and constVal != -1.0)
                                continue;

                            if (opName == "MatrixMul" or opName == "Mul") {
                                // mul(1,x) and mul(-1,x)
                                conNode.remove_child(node);
                                node.replace_const_eli(constVal, nConNode); //const value
                            }
                            else if (opName == "Pow" and constVal == 1.0) {
                                // pow(1,x)
                                conNode.remove_child(node);
                                node.replace_const_eli(1, nConNode); //const value
                            }

                            if (conNode->children.empty())
                                conNode->active = false;
                        }
                    }
                    // To-do: for mul(1, x, y) -> mul(x, y)
                }

                _graph->removeInactiveNodes();
            }
            
        };
    }
}
#endif //METADIFF_OPT_OPTIMIZER_H