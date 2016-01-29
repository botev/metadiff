//
// Created by alex on 16/01/16.
//

#ifndef METADIFF_CORE_IMPL_H
#define METADIFF_CORE_IMPL_H

namespace metadiff{

    void Node::copy_to(GraphInPtr graph, std::vector<Node> ancestors){
        std::shared_ptr<NodeInternal> ptr = unwrap();
        std::shared_ptr<NodeInternal> node = std::make_shared<NodeInternal>(graph, ptr->device);
        node->id = graph->nodes.size();
        graph->nodes.push_back(node);
        node->device = ptr->device;
        node->name = ptr->name;
        node->type = ptr->type;
        node->v_type = ptr->v_type;
        node->shape = ptr->shape;
        node->op = ptr->op->copy_to(graph, ancestors);
        node->grad_level = ptr->grad_level;
        node->value = ptr->value;
        node->shared = ptr->shared;
        node->execution = ptr->execution;
        for(size_t i=0;i<ancestors.size();i++){
            ancestors[i].unwrap()->children.push_back(node);
        }
    }

//    void Node::update_grad_level(){
//        if(ptr->id == ptr->graph->nodes.size()-1){
//            NodeVec parents = ptr->op->get_parents();
//            for(int i=0;i<parents.size();i++){
//                if(ptr->grad_level < parents[i].unwrap()->grad_level){
//                    ptr->grad_level = parents[i].unwrap()->grad_level;
//                }
//            }
//        }
//    }

    void Node::update(Node update){
        std::shared_ptr<NodeInternal> ptr = unwrap();
        ptr->graph->update_node(Node(ptr), update);
    }

    bool Node::is_constant() const{
        std::shared_ptr<NodeInternal> ptr = unwrap();
        for(int i=0;i<ptr->graph->temporary_constants.size(); i++){
            if(ptr->graph->temporary_constants[i].unwrap() == ptr){
                return true;
            }
        }
        if(ptr->type == CONSTANT or ptr->type == CONSTANT_DERIVED
           or ptr->type == SYMBOLIC_INTEGER or ptr->type == UPDATE){
            return true;
        } else {
            return false;
        }
    }

    bool Node::is_scalar() const{
        std::shared_ptr<NodeInternal> ptr = unwrap();
        for(int i=0; i < 4; i++){
            if(ptr->shape[i] != 1){
                return false;
            }
        }
        return true;
    }

    bool Node::is_vector() const{
        std::shared_ptr<NodeInternal> ptr = unwrap();
        for(int i=1; i < 4; i++){
            if(ptr->shape[i] != 1){
                return false;
            }
        }
        return true;
    }

    bool Node::is_vector_strict() const{
        std::shared_ptr<NodeInternal> ptr = unwrap();
        for(int i=0; i < 1; i++){
            if(ptr->shape[i] == 1){
                return false;
            }
        }
        for(int i=1; i < 4; i++){
            if(ptr->shape[i] != 1){
                return false;
            }
        }
        return true;
    }

    bool Node::is_matrix() const{
        std::shared_ptr<NodeInternal> ptr = unwrap();
        for(int i=2; i < 4; i++){
            if(ptr->shape[i] != 1){
                return false;
            }
        }
        return true;
    }

    bool Node::is_matrix_strict() const{
        std::shared_ptr<NodeInternal> ptr = unwrap();
        for(int i=0; i < 2; i++){
            if(ptr->shape[i] == 1){
                return false;
            }
        }
        for(int i=2; i < 4; i++){
            if(ptr->shape[i] != 1){
                return false;
            }
        }
        return true;
    }

    bool Node::is_tensor3() const{
        std::shared_ptr<NodeInternal> ptr = unwrap();
        for(int i=3; i < 4; i++){
            if(ptr->shape[i] != 1){
                return false;
            }
        }
        return true;
    }

    bool Node::is_tensor3_strict() const{
        std::shared_ptr<NodeInternal> ptr = unwrap();
        for(int i=0; i < 3; i++){
            if(ptr->shape[i] == 1){
                return false;
            }
        }
        for(int i=3; i < 4; i++){
            if(ptr->shape[i] != 1){
                return false;
            }
        }
        return true;
    }

    bool Node::is_tensor4_strict() const{
        std::shared_ptr<NodeInternal> ptr = unwrap();
        for(int i=0; i < 4; i++){
            if(ptr->shape[i] == 1){
                return false;
            }
        }
        return true;
    }

    template <typename T>
    Node apply(Node node) {
        return node.unwrap()->graph->derived_node(std::make_shared<T>(node.unwrap()->graph, node));
    }


    template <typename T>
    Node apply(Node parent1, Node parent2){
        GraphInPtr graph = parent1.unwrap()->graph;
        return graph->derived_node(std::make_shared<T>(graph, parent1, parent2));
    }

    template <typename T>
    Node apply(NodeVec parents){
        GraphInPtr graph = parents[0].unwrap()->graph;
        return graph->derived_node(std::make_shared<T>(graph, parents));
    }

    void Operator::generate_gradients(std::vector<Node> &messages){
        // Check for any incoming messages
        if(messages[owner.unwrap()->id].empty()){
            return;
        }

        // Get the gradient with respect to this node
        Node my_grad = messages[owner.unwrap()->id];
        // Update the message name
        if(my_grad.unwrap()->name == "Derived Node" or my_grad.unwrap()->name == ""){
            my_grad.unwrap()->name = "Grad of " + std::to_string(owner.unwrap()->id) + "|";
        } else {
            my_grad.unwrap()->name += "Grad of " + std::to_string(owner.unwrap()->id) + "|";
        }

        // Check for any surprises, where all parents are constants
        // If that is the case this node should have been constant as well
        // and no message should have been sent to it
        NodeVec parents = get_parents();
        bool constant = not (owner.unwrap()->op->name == "Input");
        for(int i=0;i<parents.size();i++){
            if(not parents[i].is_constant()){
                constant = false;
                break;
            }
        }
        if(constant){
            throw UnknownError({parents}, "Gradient message present, all parents are constants");
        }

        // Compute and send gradients only to non constant nodes and send them
        for(size_t i=0;i<parents.size();i++) {
            if(not parents[i].is_constant()) {
                Node parent_grad = get_parent_grad(my_grad, i);
                if(parent_grad.unwrap()->name == "Derived Node" or parent_grad.unwrap()->name == ""){
                    parent_grad.unwrap()->name = "Grad msg " + std::to_string(owner.unwrap()->id) + "->"
                                            + std::to_string(parents[i].unwrap()->id) + "|";
                } else {
                    parent_grad.unwrap()->name += "Grad msg " + std::to_string(owner.unwrap()->id) + "->"
                                             + std::to_string(parents[i].unwrap()->id) + "|";
                }
                send_grad_message(parents[i].unwrap()->id, parent_grad, messages);
            }
        }
    };

    Graph create_graph(){
        return std::make_shared<GraphInternal>();
    }

    NodeVec GraphInternal::copy(GraphInPtr new_graph, std::vector<bool> mask){
        new_graph->name = name + "_copy";
        new_graph->default_device = default_device;
        new_graph->f_type = f_type;
        new_graph->i_type = i_type;
        new_graph->broadcast = broadcast;
        new_graph->sym_integer_count = sym_integer_count;
        new_graph->shared_vars = shared_vars;
        size_t n = nodes.size();
        NodeVec mapping(n, Node());
        for(int i=0;i<n;i++){
            if(mask[i]){
                NodeVec ancestors = nodes[i]->op->get_ancestors();
                NodeVec new_ancestors;
                for(int j=0;j<ancestors.size();j++){
                    new_ancestors.push_back(mapping[ancestors[j].unwrap()->id]);
                }
                Node(nodes[i]).copy_to(new_graph, new_ancestors);
                mapping[nodes[i]->id] = new_graph->nodes.back();
            }
        }
        // Update all names containing numbers
        // Not supported because gcc 4.8.x does not support regex
//        std::regex ids("\\s[[:digit:]]+(?=[\\|-])");
//        std::vector<size_t> positions, lengths;
//        for(size_t i=0;i<new_graph->nodes.size();i++){
//            if(new_graph->nodes[i]->grad_level > 0) {
//                Node node = new_graph->nodes[i];
//                for(std::sregex_iterator s = std::sregex_iterator(node.unwrap()->name.begin(), node.unwrap()->name.end(), ids);
//                    s != std::sregex_iterator();
//                    ++s )
//                {
//                    std::smatch m = *s;
//                    positions.push_back(m.position());
//                    lengths.push_back(m.length());
//                }
//                int diff = 0;
//                for(size_t s=0;s<positions.size();s++){
//                    std::string new_id = " " + std::to_string(mapping[node.unwrap()->id].unwrap()->id);
//                    node.unwrap()->name.replace(positions[i]+diff, lengths[i], new_id);
//                    diff += new_id.length()  - lengths[i];
//                }
//                positions.clear();
//                lengths.clear();
//            }
//        }
        return mapping;
    }


    std::vector<bool> GraphInternal::get_descendants_mask(std::vector<Node>& marked){
        auto n = nodes.size();
        std::vector<bool> descendants_mask(n, false);
        for(int i=0;i<marked.size();i++){
            descendants_mask[marked[i].unwrap()->id] = true;
        }

        // Mark all direct children
        for(int i=0;i<n; i++){
            if(descendants_mask[i]){
                auto children = nodes[i]->children;
                for(int j=0;j<children.size();j++){
                    descendants_mask[children[j].unwrap()->id] = true;
                }
            }
        }
        return descendants_mask;
    };

    std::vector<bool> GraphInternal::get_ancestors_mask(std::vector<Node>& marked){
        // Assumes that computations are ordered
        auto n = nodes.size();
        std::vector<bool> ancestors_mask(n, false);
        for(int i=0;i<marked.size();i++){
            ancestors_mask[marked[i].unwrap()->id] = true;
        }
        for(size_t i=0;i<temporary_updates.size();i++){
            ancestors_mask[temporary_updates[i].unwrap()->op->get_parents()[0].unwrap()->id] = true;
        }
        // Mark all direct ancestors
        for(size_t i=n-1;i < n; i--){
            if(ancestors_mask[i]){
                NodeVec ancestors = nodes[i]->op->get_ancestors();
                for(int j=0;j<ancestors.size();j++){
                    ancestors_mask[ancestors[j].unwrap()->id] = true;
                }
            }
        }
        return ancestors_mask;
    };

    Node GraphInternal::find_same_node(std::shared_ptr<Operator> op){
        // For the moment only one such for the experiment
        if(op->name == "Exp"){
            Node parent = op->get_parents()[0];
            for(int i=0;i<parent.unwrap()->children.size(); i++){
                Node child = parent.unwrap()->children[i];
                if(child.unwrap()->op->name == "Exp"){
                    return child.alias();
                }
            }
            for(int i=0;i<parent.unwrap()->children.size(); i++){
                Node child = parent.unwrap()->children[i];
                if(child.unwrap()->op->name == "Neg"){
                    for(int j=0;j<child.unwrap()->children.size();j++){
                        Node neg_child = child.unwrap()->children[j];
                        if(neg_child.unwrap()->op->name == "Exp"){
                            return neg_child.div();
                        }
                    }
                }
            }
            if(parent.unwrap()->op->name == "Neg"){
                Node gparent = parent.unwrap()->op->get_parents()[0];
                for(int i=0;i<gparent.unwrap()->children.size(); i++){
                    Node child = gparent.unwrap()->children[i];
                    if(child.unwrap()->op->name == "Exp"){
                        return child.div();
                    }
                }
            }
        }
        return Node();
    };

    void GraphInternal::add_temporary_updates(const Updates& updates){
        for(int i=0;i<updates.size();i++){
            temporary_updates.push_back(update_node (updates[i].first, updates[i].second));
        }
    };

    void GraphInternal::clear_temporary_updates(){
        for(int i=0;i<temporary_updates.size();i++){
            nodes.pop_back();
        }
        temporary_updates.clear();
    }

    std::vector<Node> GraphInternal::gradient(Node objective, std::vector<Node> params){
        if(not objective.is_scalar()){
            throw UnsupportedGradient();
        }
        std::vector<Node> grad_messages(nodes.size(), Node());
        // Extract the flow tree between params and objective
        std::vector<bool> descendants_mask = get_descendants_mask(params);
        NodeVec vec = {objective};
        std::vector<bool> ancestors_mask = get_ancestors_mask(vec);
        std::vector<Node> flow_tree;
        temporary_constants.clear();
        for(size_t i=0;i<nodes.size(); i++) {
            if(ancestors_mask[i] and descendants_mask[i]){
                flow_tree.push_back(nodes[i]);
            } else {
                temporary_constants.push_back(nodes[i]);
            }
        }
        gradient_mode = objective.unwrap()->grad_level + 1;
        // Send the first message as 1 to the objective
        Node unity_grad = constant_value(1.0);
        grad_messages[objective.unwrap()->id] = unity_grad;
        // Send all gradient messages
        for (size_t i = flow_tree.size(); i > 0; i--) {
            if (not grad_messages[flow_tree[i-1].unwrap()->id].empty()) {
                flow_tree[i-1].unwrap()->op->generate_gradients(grad_messages);
            }
        }
        gradient_mode = 0;
        // Extract the gradients for each parameter
        std::vector<Node> grads;
        for (int i = 0; i < params.size(); i++) {
            grads.push_back(grad_messages[params[i].unwrap()->id]);
        }
        // Restore types of other inputs
        temporary_constants.clear();
        return grads;
    };

    Graph GraphInternal::optimize(NodeVec& targets, Updates& updates, NodeVec& inputs,
                                  NodeVec& new_targets, Updates& new_updates, NodeVec& new_inputs){
        // Copy only the relevant part of the graph
        Graph copy = create_graph();
        add_temporary_updates(updates);
        NodeVec mapping = this->copy(copy.get(), get_ancestors_mask(targets));
        clear_temporary_updates();
        // Optimize
        for(size_t i=0;i<copy->nodes.size();i++){
            Node node = copy->nodes[i];
            if(node.is_scalar() and node.is_constant()){
                node.unwrap()->execution.inlined = true;
            }
            if(node.unwrap()->op->name == "Broadcast"){
                node.unwrap()->execution.inlined = true;
            }
            if(node.unwrap()->op->name == "Transpose"){
                node.unwrap()->execution.inlined = true;
            }
            if(node.unwrap()->op->name == "Neg"){
                node.unwrap()->execution.inlined = true;
            }
            if(node.unwrap()->children.size() <= 1){
                node.unwrap()->execution.inlined = true;
            }
        }
        // Set the new_targets and new_updates
        for(int i=0;i<targets.size();i++){
            new_targets.push_back(mapping[targets[i].unwrap()->id]);
        }
        for(int i=0;i<updates.size();i++){
            Node node1 = mapping[updates[i].first.unwrap()->id];
            Node node2 = mapping[updates[i].second.unwrap()->id];
            new_updates.push_back(std::pair<Node, Node>(node1, node2));
        }
        for(int i=0;i<inputs.size();i++){
            new_inputs.push_back(mapping[inputs[i].unwrap()->id]);
        }
        return copy;
    };

    Node GraphInternal::shared_var(af::array value, std::string name){
        ad_value_type dtype;
        if(value.type() == af::dtype::b8){
            dtype = BOOLEAN;
        } else if(value.type() == af::dtype::f32
                  or value.type() == af::dtype::f64){
            dtype = FLOAT;
        } else {
            dtype = INTEGER;
        }
        af::dim4 dims = value.dims();
        std::shared_ptr<NodeInternal> result = std::make_shared<NodeInternal>(
                shared_from_this().get(),
                default_device,
                nodes.size(),
                name,
                ad_node_type::SHARED_INPUT,
                dtype,
                Shape {dims[0], dims[1], dims[2], dims[3]},
                std::make_shared<Input>(shared_from_this().get()),
                0
        );
        result->shared = std::make_shared<SharedVariable>(shared_vars.size(), value);
        shared_vars.push_back(result->shared);
        nodes.push_back(result);
        result->op->owner = result;
        return result;
    };

    Node GraphInternal::derived_node(std::shared_ptr<Operator> op, size_t grad_level){
        Node same_node = find_same_node(op);
        if(same_node.empty()) {
            grad_level = gradient_mode > op->get_gradient_level() ? gradient_mode : op->get_gradient_level();
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this().get(),
                    default_device,
                    nodes.size(),
                    "Derived Node",
                    op->get_node_type(),
                    op->get_value_type(),
                    op->get_shape(),
                    op,
                    grad_level
            );
            nodes.push_back(result);
            op->owner = result;
            NodeVec ancestors = op->get_ancestors();
            for (int i = 0; i < ancestors.size(); i++) {
                ancestors[i].unwrap()->children.push_back(result);
            }
            return result;
        } else {
            return same_node;
        }
    }

    Node GraphInternal::update_node(Node shared,
                                    Node update,
                                    size_t grad_level) {
        auto op = std::make_shared<Update>(shared_from_this().get(), shared, update);
        Node same_node = find_same_node(op);
        if(same_node.empty()) {
            grad_level = grad_level == GRAD_LEVEL_BAR ? op->get_gradient_level() : grad_level;
            auto result = std::make_shared<NodeInternal>(
                    shared_from_this().get(),
                    default_device,
                    nodes.size(),
                    "Update Node",
                    op->get_node_type(),
                    op->get_value_type(),
                    op->get_shape(),
                    op,
                    grad_level
            );
            nodes.push_back(result);
            op->owner = result;
            NodeVec ancestors = op->get_ancestors();
            for (int i = 0; i < ancestors.size(); i++) {
                ancestors[i].unwrap()->children.push_back(result);
            }
            return result;
        } else {
            return same_node;
        }
    };

    Node GraphInternal::constant_node(af::array value){
        ad_value_type dtype;
        if(value.type() == af::dtype::b8){
            dtype = BOOLEAN;
        } else if(value.type() == af::dtype::f32
                  or value.type() == af::dtype::f64){
            dtype = FLOAT;
        } else {
            dtype = INTEGER;
        }
        af::dim4 dims = value.dims();
        std::shared_ptr<NodeInternal> result = std::make_shared<NodeInternal>(
                shared_from_this().get(),
                default_device,
                nodes.size(),
                "Constant Node",
                ad_node_type::CONSTANT,
                dtype,
                Shape {dims[0], dims[1], dims[2], dims[3]},
                std::make_shared<Input>(shared_from_this().get()),
                0
        );
        result->value = value;
        nodes.push_back(result);
        result->op->owner = result;
        return result;
    };

    Node GraphInternal::tensor(ad_value_type v_type,
                               std::array<SymInt, 4> shape,
                               std::string name) {
        auto result = std::make_shared<NodeInternal>(
                shared_from_this().get(),
                default_device,
                nodes.size(),
                name,
                ad_node_type::INPUT,
                v_type,
                shape,
                std::make_shared<Input>(shared_from_this().get()),
                0
        );
        nodes.push_back(result);
        result->op->owner = result;
        return result;
    }

    Node GraphInternal::tensor(ad_value_type v_type,
                               SymInt shape0,
                               SymInt shape1,
                               SymInt shape2,
                               SymInt shape3,
                               std::string name) {
        std::array<SymInt, 4> shape{shape0,
                                    shape1,
                                    shape2,
                                    shape3};
        return tensor(v_type, shape, name);
    }

    Node GraphInternal::tensor(ad_value_type v_type,
                               std::string name) {
        std::array<SymInt, 4> shape = {
                get_new_symbolic_integer(),
                get_new_symbolic_integer(),
                get_new_symbolic_integer(),
                get_new_symbolic_integer()
        };
        return tensor(v_type, shape, name);
    }

    Node GraphInternal::tensor_as(Node node, std::string name) {
        return tensor(node.unwrap()->v_type, node.unwrap()->shape, name);
    }

    Node GraphInternal::tensor3(ad_value_type v_type,
                                std::array<SymInt, 3> shape,
                                std::string name) {
        return tensor(v_type, {shape[0], shape[1], shape[2], 1}, name);
    }

    Node GraphInternal::tensor3(ad_value_type v_type,
                                SymInt shape0,
                                SymInt shape1,
                                SymInt shape2,
                                std::string name) {
        return tensor(v_type, std::array<SymInt, 4>{
                              shape0,
                              shape1,
                              shape2,
                              1
                      },
                      name);
    }

    Node GraphInternal::tensor3(ad_value_type v_type,
                                std::string name) {
        auto shape0 = get_new_symbolic_integer();
        auto shape1 = get_new_symbolic_integer();
        auto shape2 = get_new_symbolic_integer();
        return tensor3(v_type, shape0, shape1, shape2, name);
    }


    Node GraphInternal::tensor3_as(Node node, std::string name) {
        if(not node.is_tensor3()){
            throw "Node with id '" + std::to_string(node.unwrap()->id) + "' is not a tensor3.";
        }
        return tensor3(nodes[node.unwrap()->id]->v_type,
                       nodes[node.unwrap()->id]->shape[0],
                       nodes[node.unwrap()->id]->shape[1],
                       nodes[node.unwrap()->id]->shape[2],
                       name);
    }

    Node GraphInternal::matrix(ad_value_type v_type,
                               std::array<SymInt, 2> shape,
                               std::string name) {
        std::array<SymInt, 4> shape_t{shape[0], shape[1], SymInt::one(), SymInt::one()};
        return tensor(v_type, shape_t, name);
    }

    Node GraphInternal::matrix(ad_value_type v_type,
                               SymInt shape0,
                               SymInt shape1,
                               std::string name) {
        return tensor(v_type, std::array<SymInt, 4>{
                              shape0,
                              shape1,
                              SymInt::one(),
                              SymInt::one()},
                      name);
    }

    Node GraphInternal::matrix(ad_value_type v_type,
                               std::string name) {
        auto shape0 = get_new_symbolic_integer();
        auto shape1 = get_new_symbolic_integer();
        return matrix(v_type, shape0, shape1, name);
    }


    Node GraphInternal::matrix_as(Node node, std::string name) {
        if(not node.is_matrix()){
            throw "Node with id '" + std::to_string(node.unwrap()->id) + "' is not a matrix.";
        }
        return matrix(nodes[node.unwrap()->id]->v_type,
                      nodes[node.unwrap()->id]->shape[0],
                      nodes[node.unwrap()->id]->shape[1],
                      name);
    }

    Node GraphInternal::square_matrix(ad_value_type v_type,
                                      SymInt shape,
                                      std::string name) {
        std::array<SymInt, 4> shape_t{shape, shape, SymInt::one(), SymInt::one()};
        return tensor(v_type, shape_t, name);
    }

    Node GraphInternal::vector(ad_value_type v_type,
                               SymInt shape,
                               std::string name) {
        std::array<SymInt, 4> shape_t{shape, SymInt::one(), SymInt::one(), SymInt::one()};
        return tensor(v_type, shape_t, name);
    }

    Node GraphInternal::vector(ad_value_type v_type,
                               std::string name) {
        auto shape0 = get_new_symbolic_integer();
        return vector(v_type, shape0, name);
    }


    Node GraphInternal::vector_as(Node node,
                                  std::string name) {
        if(not node.is_vector()){
            throw "Node with id '" + std::to_string(node.unwrap()->id) + "' is not a vector.";
        }
        return vector(nodes[node.unwrap()->id]->v_type,
                      nodes[node.unwrap()->id]->shape[0],
                      name);
    }

    Node GraphInternal::scalar(ad_value_type v_type,
                               std::string name) {
        std::array<SymInt, 4> shape_t{SymInt::one(), SymInt::one(), SymInt::one(), SymInt::one()};
        return tensor(v_type, shape_t, name);
    }

}
#endif //METADIFF_CORE_IMPL_H
