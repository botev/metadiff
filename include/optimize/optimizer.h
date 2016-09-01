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
    *   this is to experiment new graph optimizations only
    */
    Optimizer(shared_ptr<GraphInternal> graph) : _graph(graph), _lastNodeId(graph->nodes.back()->id) {
        logger()->debug() << "Experiment optimization";
    }

    /**
    *   run optimization based on given option
    */
    void run(Option option=Option::FAST) {
        logger()->debug() << "optimizating graph with option " << option;

        opt_filter_nodes();

        if (option != Option::FAST) {
            size_t lastId = _lastNodeId;
            size_t lastSize = _graph->nodes.size();
            // hard code iterations
            for (size_t i=0; i<10; i++) {
                logger()->debug() << "Iteration " << i+1;

                // opt_merge();
                // opt_const_folding();
                // opt_const_elimination();
                // opt_neg_neg();
                // opt_add_zero();
                // opt_sum_scalar_mul();

                //remove all inactive nodes from graph
                _graph->removeInactiveNodes();

                if (lastSize == _graph->nodes.size() and lastId == _graph->nodes.back()->id)
                    break;
                lastId = _graph->nodes.back()->id;
                lastSize = _graph->nodes.size();
            }

            const bool noNewNode = _lastNodeId >= _graph->nodes.back()->id ? true : false;
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
        }

        opt_inline();
        opt_elementwise_inplace();
        // opt_planning(); buggy...

        logger()->debug() << "optimization finished";
    }

private:
    // use this id to track if new node is added in optimization
    size_t _lastNodeId;
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

        _lastNodeId = _graph->nodes.back()->id;
    }


    /**
     * filter out irrelevant nodes in graph based on dfs from _targets and _updates
    */
    void opt_filter_nodes() {

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

    /**
    *   merge indentical pattern in graph
    *   If two nodes have identical inputs and operator, they are considered identical
    */
    void opt_merge() {
        logger()->debug() << "opt_merge...";
        // key:operator name; value:node
        typedef unordered_map<string, Node> opMap;
        // key:parents ids
        map<vector<int>, opMap> nodeMap;

        for(Node node : _graph->nodes) {
            if(!node.is_active()) continue;

            vector<int> parentIds{};
            node->op->get_parents_ids(parentIds);
            if (parentIds.empty()) {
                // consider merge constant values 
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

    /** helper method for opt_const_folding */
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
        if (node.is("Add") or node.is("Mul")) {
            double value = node.is("Add") ? 0.0 : 1.0;
            for (Node p : parents) {

                if(p.is_active()) {
                    if (node.is("Add"))
                        value += p->op->getConstVal();
                    else
                        value *= p->op->getConstVal();
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

    /**
    *   If inputs of a nodes are all constants, compute the value and replace the node
    */
    void opt_const_folding() {
        logger()->debug() << "opt_const_folding...";

        // ids of nodes
        unordered_set<int> visited;
        visited.reserve(_graph->nodes.size());

        for(Node node : _graph->nodes) {
            if(!node.is_active()) continue;

            const_folding_dfs(node, visited);
        }
    }

    /**
    *    eliminate constants in mul
    */
    void opt_const_elimination() {
        logger()->debug() << "opt_const_elimination...";

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
                    // else if (node.is("Pow")) {
                    //     if (constVal == 0.0) {
                    //         // pow(0, x)
                    //     }
                    //     else if (constVal == 1.0) {
                    //         // pow(1,x)
                    //         conNode.remove_child(node);
                    //         node.replace_const_eli(1, nConNode); //const value
                    //     }
                    //     else if (constVal == -0.5) {
                    //         // pow(x, -0.5) -> inv(sqrt(x))
                    //     }
                    // }

                    if (conNode->children.empty())
                        conNode.set_inactive();
                }
            }
            // To-do: for mul(1, x, y) -> mul(x, y)
        }
    }

    /** eliminate double neg() calls on a node */
    void opt_neg_neg() {
        logger()->debug() << "opt_neg_neg...";
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

    /**
    *   sum(scalar * tensor) -> scalar * sum(tensor)
    */
    void opt_sum_scalar_mul() {
        logger()->debug() << "opt_sum_scalar_mul...";

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

            NodeVec actives{}; 
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
                    // all other children of ancestor have id smaller than this inplace child
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
    *   naive memory planning
    */
    void opt_planning() {
        logger()->debug() << "opt_planning...";
        auto pool = std::make_shared<TagPool>();

        // prepare initial queue for bfs
        unordered_set<int> visited;
        std::list<Node> bfsq;
        for(Node node : _graph->nodes) {

            if (node.is("Input") or node.is("Shared") or node.is_constant()) {
                node->execution.inlined = true;
                for (auto c : node->children)
                    bfsq.push_back(c);
            }
        }

        while (!bfsq.empty()) {
            Node current = bfsq.front();
            bfsq.pop_front();

            if(visited.find(current->id) != visited.end()) continue;
            visited.insert(current->id);

            pool->allocate(current);
            cout<<"allocate "<< current->execution.tag<<" "<< current->id<<endl;

            if (current->children.size() == 1) {
                // inplace
                current->children[0]->execution.tag = current->execution.tag;
                bfsq.push_front(current->children[0]);
            }
            else {
                // memory sharing
                for (int i=0; i<current->children.size(); i++) {
                    bfsq.push_front(current->children[i]);
                    pool->allocate(current->children[i]);
                    cout<<"allocateC "<< current->children[i]->execution.tag<<" "<< current->children[i]->id<<endl;
                }
                // release the tag back to pool after all immediate children have been allocated a tag
                pool->push(current->execution.tag);
            }
        }
    }

    class TagPool {
    public:
        TagPool(): _counter(0), _pool(){}
        inline int top() {return _pool.top();}
        inline void pop() {_pool.pop();}
        inline void push(int x) {_pool.push(x);}

        /**
        *   allocate a tag to a node
        */
        void allocate(Node node) {

            if (node->execution.tag != -1) return;

            if (_pool.empty()) {
                node->execution.tag = _counter++;
            }
            else {
                // from pool
                node->execution.tag = _pool.top();
                _pool.pop();
            }
        }

    private:
        int _counter;
        stack<int> _pool;
    };

public:
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