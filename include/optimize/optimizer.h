#ifndef METADIFF_OPT_OPTIMIZER_H
#define METADIFF_OPT_OPTIMIZER_H

namespace metadiff{
    namespace opt {
        using namespace core;

/** optimization options */
enum Option {
    FAST = 0,
    FULL = 1
};

class Optimizer {

private:
    size_t _originalSize;
    // optimized graph
    shared_ptr<GraphInternal> _graph;

    // optimized new members
    NodeVec _inputs;
    NodeVec _targets;
    Updates _updates;

    inline shared_ptr<spdlog::logger> logger() const {
            return logging::logger("optimizer");
    }

    /**
    *   Default inplace optimization when single version of updates will be applied
    *   Or
    *   make deep copy and apply otimization on the copied graph
    *   when multiple versions of updates will be applied on the same graph
    *   in this case, new targets, updates, and inputs are stored in data members
    */
    Optimizer(shared_ptr<GraphInternal> graph, NodeVec inputs, NodeVec targets, Updates updates, bool inplace)
    {
        if (inplace) {
            logger()->debug() << "Inplace optimization";
            _graph = graph;
            _originalSize = _graph->nodes.size();
            _targets = targets;
            _updates = updates;
            _inputs = inputs;
        }
        else {
            logger()->debug() << "Optimization on deep copy";
            _graph = std::make_shared<GraphInternal>();
            graph->add_temporary_updates(updates);

            NodeVec marked;
            marked.reserve(targets.size() + updates.size());
            for(auto& i : targets) marked.push_back(i);
            for(auto& u : updates) marked.push_back(u.second);

            NodeVec mapping = graph->copy(_graph.get(), graph->get_ancestors_mask(marked));
            graph->clear_temporary_updates();

            for (int i = 0; i < targets.size(); i++) {
                _targets.push_back(mapping[targets[i]->id]);
            }
            for (int i = 0; i < updates.size(); i++) {
                Node node1 = mapping[updates[i].first->id];
                Node node2 = mapping[updates[i].second->id];
                _updates.push_back(std::pair<Node, Node>(node1, node2));
            }
            for (int i = 0; i < inputs.size(); i++) {
                _inputs.push_back(mapping[inputs[i]->id]);
            }
        }
    }

public:

    /**
    *   return a unique_ptr of Optimizer object
    */
    static unique_ptr<Optimizer> create(shared_ptr<GraphInternal> graph, NodeVec inputs, NodeVec targets, Updates updates, bool inplace=false) {
            return unique_ptr<Optimizer>(new Optimizer(graph, inputs, targets, updates, inplace));
    }

    /**
    *   run the optimizer, return a unique_ptr of it
    */
    static unique_ptr<Optimizer> execute(shared_ptr<GraphInternal> graph, NodeVec inputs, NodeVec targets, Updates updates, 
        bool inplace=false, Option option=Option::FAST) {
            auto opti = create(graph, inputs, targets, updates, inplace);
            opti->run(option);
            return opti;
    }

    /**
    *  this is to experiment new graph optimizations only
    */
    Optimizer(shared_ptr<GraphInternal> graph) : _graph(graph), _originalSize(_graph->nodes.size()) {}

    void run(Option option=Option::FAST) {
        logger()->debug() << "optimizating graph with option " <<option;

        opt_filter_nodes();

        if (option != Option::FAST) {
            // opt_merge();
            // opt_const_folding();
            // opt_add_zero();
            // opt_const_elimination();
            // opt_neg_neg();
            // opt_sum_scalar_martix();
        }

        opt_inline();
        opt_elementwise_inplace();

        const bool noNewNode = _originalSize >= _graph->nodes.size() ? true : false;
        //remove all inactive nodes from graph
        _graph->removeInactiveNodes();

        // if no new node is added, a topological sort is not needed
        if (noNewNode)
        {
            for(int i=0; i<_graph->nodes.size(); i++) {
                _graph->nodes[i]->id = i;
            }
        }
        else {
            // ensure the ordering of nodes in graph
            _graph->topo_sort();
        }

        logger()->debug() << "optimization finished";
    }

    void opt_filter_nodes() {
        /**
         * filter out irrelevant nodes in graph based on dfs from _targets and _updates
        */
        logger()->debug() << "opt_filter_nodes...";
        NodeVec startNodes(_targets);
        for(auto& u : _updates)
            startNodes.push_back(u.second);

        unordered_set<shared_ptr<NodeInternal>> validNodes = _graph->get_nodes_and_ancestors(startNodes);

        for(Node n : _graph->nodes) {
            if (validNodes.find(n.unwrap()) == validNodes.end()) {
                n.set_inactive();
            }
        }
    }

    void opt_merge() {
        // logger()->debug() << "1. merge...";
        // key:operator name; value:node
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
                    else if (node.is("Pow")) {
                        if (constVal == 0.0) {
                            // pow(0, x)
                        }
                        else if (constVal == 1.0) {
                            // pow(1,x)
                            conNode.remove_child(node);
                            node.replace_const_eli(1, nConNode); //const value
                        }
                        else if (constVal == -0.5) {
                            // pow(x, -0.5) -> inv(sqrt(x))
                        }
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
            logger()->debug()<<"create mul node: "<<sMul->id;

            sMul->children = sumNodeChildren;
        }
    }

    /**
     * eliminate zero in add operator
     */
    void opt_add_zero() {
        for(Node node : _graph->nodes) {
            if(!node.is_active() or !node.is("Add"))
                continue;

            NodeVec actives; 
            for(Node ance : node->op->get_parents()) {
                if (ance.is_constant() and ance->op->getConstVal() == 0.0) {
                    ance.set_inactive();
                }
                else {
                    actives.push_back(ance);
                }
            }
            node->op->update_parents(actives);
            int parentsSize = node->op->get_parents().size();
            
            if (parentsSize == 0) {
                // it is now a node of constant zero
                node.replace_with_constant(0.0);
            }
            else if (parentsSize == 1) {
                // bypass current Add node
                node.set_inactive();
                node->op->get_parents()[0].replace_children_from(node);
                node.replace_parent_of_children(node->op->get_parents()[0]);
            }
        }
    }

    void opt_canonize() {

    }

    

    /**
     * populating execution data - inlined
     * the actual inline checks are done during source generation
    */
    void opt_inline() {
        logger()->debug() << "opt_inline...";
        const vector<string> ops{"Input", "Shared", "Broadcast", "Transpose", "Neg"};

        for (Node node : _graph->nodes) {
            if(!node.is_active()) continue;

            if (node.is_scalar() and node.is_constant()) {
                node->execution.inlined = true;
            }
            else if (node->children.size() <= 1) {
                node->execution.inlined = true;
            }
            else if(std::find(ops.begin(), ops.end(), node->op->name) != ops.end()) {
                node->execution.inlined = true;
            }
        }
    }

    /**
     *  if one of the inputs of an elementwise operator has same type and shape to the output
     *  and is no longer useful after the elementwise operator
     *  reuse the storage of input as storage of output

     *  populating execution data - inplace
     *  the actual inline checks are done during source generation
     *  this optimization should be done after opt_inline
    */
    void opt_elementwise_inplace() {
        logger()->debug() << "opt_elementwise_inplace...";
        for (Node node : _graph->nodes) {
            if(!node.is_active()) continue;

            if (node->op->is_elementwise()) {
                //O(ancestors*children), but there won't be many of them)
                for(Node ancestor : node->op->get_ancestors()) {

                    // need a proper logic for it. could be:
                    // 1. topo_sort (not done properly yet)
                    // 2. all other children of ancestor have id smaller than this inplace child
                    // only if it is no longer useful
                    bool onlyChild = true;
                    for (Node c : ancestor->children) {
                        if (c.unwrap() != node.unwrap()) {
                            onlyChild = false;
                            break;
                        }
                    }
                    if (!onlyChild) continue;

                    for(Node child : node->children) {
                        if (ancestor->shape == child->shape and 
                            ancestor->dtype == child->dtype and
                            !ancestor->execution.inlined) {
                            // if the parent is already inlined
                            // inplace from it has no gain

                            // copy inplace reference or set to the ancestor
                            if (ancestor->execution.inplace) {
                                child->execution.inplace = ancestor->execution.inplace;
                            }
                            else {
                                child->execution.inplace = ancestor.unwrap();
                            }
                        }
                    }
                }
            }
        }
    }


    /**
    *   inline getter of optimized graph
    *   it is as same as argument input when inplace optimization is done
    */
    inline shared_ptr<GraphInternal> getGraph() const {
        return _graph;
    }

    /**
    *   inline getter of optimized inputs
    *   it is as same as argument input when inplace optimization is done
    */
    inline NodeVec getInputs() const {
        return _inputs;
    }

    /**
    *   inline getter of optimized targets
    *   it is as same as argument input when inplace optimization is done
    */
    inline NodeVec getTargets() const {
        return _targets;
    }

    /**
    *   inline getter of optimized updates
    *   it is as same as argument input when inplace optimization is done
    */
    inline Updates getUpdates() const {
        return _updates;
    }

};

}}
#endif //METADIFF_OPT_OPTIMIZER_H