//
// Created by alex on 16/01/16.
//

#ifndef METADIFF_CORE_IMPL_H
#define METADIFF_CORE_IMPL_H

namespace metadiff{
    namespace core {
        using namespace exceptions;

        void operate_policy(errorPolicy policy,
                            std::shared_ptr<spdlog::logger> const logger,
                            GraphError const & err){
            switch (policy){
                case QUIET: return;
                case WARN: {
                    logger->warn() << err.msg;
                    return;
                }
                default: {
                    logger->error() << err.msg;
                    throw err;
                }
            }
        }

//        dType convert_af_dtype(af_dtype dtype){
//            switch (dtype){
//                case af_dtype::b8 : return core::b8;
//                case af_dtype::u8 : return core::u8;
//                case af_dtype::u16: return core::u16;
//                case af_dtype::u32: return core::u32;
//                case af_dtype::u64: return core::u64;
//                case af_dtype::s16: return core::i16;
//                case af_dtype::s32: return core::i32;
//                case af_dtype::s64: return core::i64;
//                case af_dtype::f32 : return core::f32;
//                case af_dtype::f64: return core::f64;
//                default: throw 20;
//            }
//        }

        dType default_dType_promotion(dType type1,
                                      dType type2,
                                      dType max_float,
                                      dType max_int){
            if(type1 == type2){
                return type1;
            } else if(type1 == f64 or type2 == f64){
                return max_float;
            } else if(type1 == f32 or type2 == f32){
                if((type1 == i64 or type2 == i64) and max_float == f64){
                    return f64;
                } else if(max_float == f64){
                    return f32;
                } else {
                    return max_float;
                }
            } else if(type1 == f16 or type2 == f16){
                if((type1 == i64 or type2 == i64) and max_float == f64) {
                    return f64;
                } else if((type1 == i32 or type2 == i32) and (max_float == f64 or max_float == f32)) {
                    return f32;
                } else if(max_float == f64 or max_float == f32){
                    return f16;
                } else {
                    return max_float;
                }
            } else if(type1 == f8 or type2 == f8){
                if((type1 == i64 or type2 == i64) and max_float == f64) {
                    return f64;
                } else if((type1 == i32 or type2 == i32) and (max_float == f64 or max_float == f32)) {
                    return f32;
                } else if((type1 == i16 or type2 == i16) and (max_float != f8)) {
                    return f16;
                } else {
                    return f8;
                }
            } else if(type1 == i64 or type2 == i64){
                return max_int;
            } else if(type1 == i32 or type2 == i32){
                if(max_int == i64){
                    return i32;
                } else {
                    return max_int;
                }
            } else if(type1 == i16 or type2 == i16){
                if(max_int == i64 or max_int == i32){
                    return i16;
                } else {
                    return max_int;
                }
            } else if(type1 == i8 or type2 == i8){
                return i8;
            } else{
                return b8;
            }
        }

        std::shared_ptr<spdlog::logger> Node::logger() const {
            return logging::logger(unwrap()->graph->name + "::node::" + std::to_string(unwrap()->id));
        }

        std::shared_ptr<NodeInternal> Node::unwrap() const {
            if (ptr.expired()) {
                logging::logger("XXX::node::XXX")->error() << "Trying to access a Node whose pointer has expired";
                exit(1);
            }
            return ptr.lock();
        }

        std::shared_ptr<NodeInternal> Node::operator->() const {
            return unwrap();
        }

        void Node::copy_to(const GraphInPtr graph, NodeVec ancestors) const {
            logger()->trace() << "Copying to node " << graph->name << "#" <<  graph->nodes.size();
            std::shared_ptr<NodeInternal> ptr = unwrap();
            std::shared_ptr<NodeInternal> node = std::make_shared<NodeInternal>(graph, ptr->device);
            node->id = graph->nodes.size();
            graph->nodes.push_back(node);
            node->device = ptr->device;
            node->name = ptr->name;
            node->node_type = ptr->node_type;
            node->dtype = ptr->dtype;
            node->shape = ptr->shape;
            node->op = ptr->op->copy_to(graph, ancestors);
            node->grad_level = ptr->grad_level;
//        node->value = ptr->value;
//        node->shared = ptr->shared;
            node->execution = ptr->execution;
            node->group = ptr->group;
            for (size_t i = 0; i < ancestors.size(); i++) {
                ancestors[i]->children.push_back(node);
            }
        }

        bool Node::is_constant() const {
            std::shared_ptr<NodeInternal> ptr = unwrap();
            if (ptr->node_type == CONSTANT or ptr->node_type == CONSTANT_DERIVED) {
                return true;
            }
            return ptr->graph->is_temporary_constant(this);
        }

        bool Node::is_scalar() const {
            std::shared_ptr<NodeInternal> ptr = unwrap();
            for (int i = 0; i < 4; i++) {
                if (ptr->shape[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        bool Node::is_vector() const {
            std::shared_ptr<NodeInternal> ptr = unwrap();
            for (int i = 1; i < 4; i++) {
                if (ptr->shape[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        bool Node::is_vector_strict() const {
            std::shared_ptr<NodeInternal> ptr = unwrap();
            for (int i = 0; i < 1; i++) {
                if (ptr->shape[i] == 1) {
                    return false;
                }
            }
            for (int i = 1; i < 4; i++) {
                if (ptr->shape[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        bool Node::is_matrix() const {
            std::shared_ptr<NodeInternal> ptr = unwrap();
            for (int i = 2; i < 4; i++) {
                if (ptr->shape[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        bool Node::is_matrix_strict() const {
            std::shared_ptr<NodeInternal> ptr = unwrap();
            for (int i = 0; i < 2; i++) {
                if (ptr->shape[i] == 1) {
                    return false;
                }
            }
            for (int i = 2; i < 4; i++) {
                if (ptr->shape[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        bool Node::is_tensor3() const {
            std::shared_ptr<NodeInternal> ptr = unwrap();
            for (int i = 3; i < 4; i++) {
                if (ptr->shape[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        bool Node::is_tensor3_strict() const {
            std::shared_ptr<NodeInternal> ptr = unwrap();
            for (int i = 0; i < 3; i++) {
                if (ptr->shape[i] == 1) {
                    return false;
                }
            }
            for (int i = 3; i < 4; i++) {
                if (ptr->shape[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        bool Node::is_tensor4_strict() const {
            std::shared_ptr<NodeInternal> ptr = unwrap();
            for (int i = 0; i < 4; i++) {
                if (ptr->shape[i] == 1) {
                    return false;
                }
            }
            return true;
        }

        void Operator::send_grad_message(size_t target, Node msg,
                                         std::vector<Node> &messages) const {
            logger()->debug() << "Sending gradient message with id "
            << msg->id << " from " <<  owner->id << " to " << target;

            if (not messages[target].ptr.expired()) {
                // If not first message add them and then send the sum
                messages[target] = Node::add(NodeVec{messages[target], msg});
            } else {
                // If first just send it
                messages[target] = msg;
            }
        }

        void Operator::generate_gradients(std::vector<Node> &messages) {
            // Check if there are any incoming gradient messages
            if (messages[owner->id].ptr.expired()) {
                return;
            }
            logger()->debug() << "Generating gradients for " << owner->id;

            // Sets the current group to _root/gradient0/parent_group
            Group current_group = graph->current_group;
            Group top_level = graph->get_group("Gradients " + std::to_string(graph->grad_level));
            Group owner_group = this->owner->group;
            std::string group_name = top_level.lock()->full_name;
            group_name += GROUP_DELIMITER;
            group_name += owner_group.lock()->full_name;
            Group grad_group = graph->get_group(group_name);
            graph->current_group = grad_group;

            // Retrieves the gradient messages representing the total gradient with respect to this node
            Node my_grad = messages[owner->id];

            // Update the gradient message name
            if (my_grad->name == "Derived Node" or my_grad->name == "") {
                my_grad->name = "Grad of " + std::to_string(owner->id) + "|";
            } else {
                my_grad->name += "Grad of " + std::to_string(owner->id) + "|";
            }

            // Check for any surprises, where all parents are constants
            // If that is the case this node should have been constant as well
            // and no message should have been sent to it, however if the node is
            // an input node than it is never constant
            NodeVec parents = get_parents();
            bool constant = name != "Input" and name != "Shared";
            for (int i = 0; i < parents.size(); i++) {
                if (not parents[i].is_constant()) {
                    constant = false;
                    break;
                }
            }
            if (constant) {
                auto err = WrongGradient(NodeVec{owner, my_grad}, name);
                logger()->error() << err.msg;
                throw err;
            }

            // Compute and send gradients only to non constant nodes
            for (size_t i = 0; i < parents.size(); i++) {
                if (not parents[i].is_constant()) {
                    Node parent_grad = get_parent_grad(my_grad, i);
                    if (parent_grad->name == "Derived Node" or parent_grad->name == "") {
                        parent_grad->name = "Grad msg " + std::to_string(owner->id) + "->"
                                            + std::to_string(parents[i]->id) + "|";
                    } else {
                        parent_grad->name += "Grad msg " + std::to_string(owner->id) + "->"
                                             + std::to_string(parents[i]->id) + "|";
                    }
                    send_grad_message(parents[i]->id, parent_grad, messages);
                }
            }
            graph->current_group = current_group;
        };

        NodeVec Operator::get_ancestors() const {
            NodeVec parents = get_parents();
            NodeVec arguments = get_arguments();
            for (int i = 0; i < arguments.size(); i++) {
                parents.push_back(arguments[i]);
            }
            return parents;
        }

        Graph create_graph() {
            return std::make_shared<GraphInternal>();
        }

        bool GraphInternal::is_temporary_constant(Node node) const {
            std::shared_ptr<NodeInternal> ptr = node.unwrap();
            for (size_t i = 0; i < temporary_constants.size(); i++) {
                if (temporary_constants[i].unwrap() == ptr) {
                    return true;
                }
            }
            return false;
        }

        NodeVec GraphInternal::copy(GraphInPtr new_graph, std::vector<bool> mask) const {
            logger()->trace() << "Copying graph " << name;
            new_graph->name = name + "_copy";
            new_graph->default_device = default_device;
            new_graph->max_float = max_float;
            new_graph->max_int = max_int;
            new_graph->promote_type = promote_type;
            new_graph->broadcast_err_policy = broadcast_err_policy;
            new_graph->type_promotion_err_policy = type_promotion_err_policy;
            new_graph->sym_integer_count = sym_integer_count;
//            new_graph->shared_vars = shared_vars;
            new_graph->groups = groups;
            size_t n = nodes.size();
            // Variable which maps each node (by id) of the original graph to a Node in the new graph
            NodeVec mapping(n, Node());
            // Copy nodes
            for (size_t i = 0; i < n; i++) {
                // Only nodes that are masked
                if (mask[i]) {
                    // Get all of the ancestors of the node and find their corresponding nodes
                    // in the new graph
                    NodeVec ancestors = nodes[i]->op->get_ancestors();
                    NodeVec new_ancestors;
                    for (size_t j = 0; j < ancestors.size(); j++) {
                        new_ancestors.push_back(mapping[ancestors[j]->id]);
                    }
                    // Copy the node using the new ancestors and put it in the mapping
                    Node(nodes[i]).copy_to(new_graph, new_ancestors);
                    mapping[nodes[i]->id] = new_graph->nodes.back();
                }
            }
            // Copy the updates, by just adding the corresponding nodes
            for (size_t i = 0; i < updates.size(); i++) {
                if (mapping[updates[i].second->id].ptr.expired()) {
                    std::cerr << "Could not copy the update for node "
                    << updates[i].first->id << " because it was not part of the mask." << std::endl;
                } else {
                    Node shared = mapping[updates[i].first->id];
                    Node update = mapping[updates[i].second->id];
                    new_graph->updates.push_back(std::pair<Node, Node>(shared, update));
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
            logger()->trace() << "Copy completed";
            return mapping;
        }

        std::vector<bool> GraphInternal::get_descendants_mask(NodeVec marked) const {
            logger()->trace() << "Generating descendants mask";
            auto n = nodes.size();
            std::vector<bool> descendants_mask(n, false);

            // Mark the nodes provided
            for (int i = 0; i < marked.size(); i++) {
                descendants_mask[marked[i]->id] = true;
            }

            // At each iteration if the node has been marked, mark its direct children
            for (int i = 0; i < n; i++) {
                if (descendants_mask[i]) {
                    auto children = nodes[i]->children;
                    for (int j = 0; j < children.size(); j++) {
                        descendants_mask[children[j]->id] = true;
                    }
                }
            }
            return descendants_mask;
        };

        std::vector<bool> GraphInternal::get_ancestors_mask(NodeVec marked) const {
            logger()->trace() << "Generating ancestors mask";
            auto n = nodes.size();
            std::vector<bool> ancestors_mask(n, false);

            // Mark the nodes provided
            for (int i = 0; i < marked.size(); i++) {
                ancestors_mask[marked[i]->id] = true;
            }

            // At each iteration if the node has been marked, mark its direct ancestors
            for (size_t i = n - 1; i < n; i--) {
                if (ancestors_mask[i]) {
                    NodeVec ancestors = nodes[i]->op->get_ancestors();
                    for (int j = 0; j < ancestors.size(); j++) {
                        ancestors_mask[ancestors[j]->id] = true;
                    }
                }
            }
            return ancestors_mask;
        };

        /**
         * TODO have to do this function properly
         */
        Node GraphInternal::find_same_node(std::shared_ptr<Operator> op) {
            if (op->get_parents().size() > 0) {
                NodeVec candidates = op->get_parents()[0]->children;
                for (int i = 0; i < candidates.size(); i++) {
                    std::shared_ptr<Operator> candidate_op = candidates[i]->op;
                    if (candidate_op->equals(op) or op->equals(candidate_op)) {
                        logger()->debug() << "Found node with id " << candidates[i]->id
                        << " to operator " << op->name;
                        return candidates[i];
                    }
                }
            }
//        NodeVec candidates = op->get_parents()
//        for(int i=0; i<op->get_parents()[0].unwrap().chilidren)
//        // For the moment only one such for the experiment
//        if(op->name == "Exp"){
//            Node parent = op->get_parents()[0];
//            for(int i=0;i<parent.unwrap()->children.size(); i++){
//                Node child = parent.unwrap()->children[i];
//                if(child.unwrap()->op->name == "Exp"){
//                    return child.alias();
//                }
//            }
//            for(int i=0;i<parent.unwrap()->children.size(); i++){
//                Node child = parent.unwrap()->children[i];
//                if(child.unwrap()->op->name == "Neg"){
//                    for(int j=0;j<child.unwrap()->children.size();j++){
//                        Node neg_child = child.unwrap()->children[j];
//                        if(neg_child.unwrap()->op->name == "Exp"){
//                            return neg_child.div();
//                        }
//                    }
//                }
//            }
//            if(parent.unwrap()->op->name == "Neg"){
//                Node gparent = parent.unwrap()->op->get_parents()[0];
//                for(int i=0;i<gparent.unwrap()->children.size(); i++){
//                    Node child = gparent.unwrap()->children[i];
//                    if(child.unwrap()->op->name == "Exp"){
//                        return child.div();
//                    }
//                }
//            }
//        }
            return Node();
        };

        void GraphInternal::add_temporary_updates(const Updates &temp_updates) {
            for (int i = 0; i < temp_updates.size(); i++) {
                logger()->trace() << "Adding a temporary update " << temp_updates[i].first->id
                << " := " << temp_updates[i].second->id;
                temporary_updates.push_back(temp_updates[i]);
            }
        };

        void GraphInternal::clear_temporary_updates() {
            // Potentially might need to sort temporary_updates
            logger()->trace() << "Clearing temporary updates";
            temporary_updates.clear();
        }

        std::vector<Node> GraphInternal::gradient(Node objective, std::vector<Node> params) {
            logger()->trace() << "Getting gradients of " << objective->id;
            // Stores the current group in order to recreate it
            Group old_group = current_group;
            if (not objective.is_scalar()) {
                auto err = UnsupportedGradient(objective);
                logger()->error() << err.msg;
                throw err;
            }

            // This vector will contain at each index the gradient message to the corresponding node
            std::vector<Node> grad_messages(nodes.size(), Node());

            // Extract the flow tree between params and objective
            std::vector<bool> descendants_mask = get_descendants_mask(params);
            std::vector<bool> ancestors_mask = get_ancestors_mask(NodeVec{objective});
            std::vector<Node> flow_tree;
            // Add all required nodes to the flow_tree and the rest to be constants
            for (size_t i = 0; i < nodes.size(); i++) {
                if (ancestors_mask[i] and descendants_mask[i]) {
                    flow_tree.push_back(nodes[i]);
                } else {
                    temporary_constants.push_back(nodes[i]);
                }
            }

            // The gradient mode is one higher than that of the objective
            grad_level = objective->grad_level + 1;

            // Set the current group to the corresponding gradients group
            current_group = get_group("Gradients " + std::to_string(grad_level));

            // Send the first message as 1 to the objective
            grad_messages[objective->id] = constant_value(1.0);

            // Send all gradient messages
            for (size_t i = flow_tree.size(); i > 0; i--) {
                if (not grad_messages[flow_tree[i - 1]->id].ptr.expired()) {
                    flow_tree[i - 1]->op->generate_gradients(grad_messages);
                }
            }

            // Reset the gradient mode
            grad_level = 0;

            // Extract the gradients for each parameter
            std::vector<Node> grads;
            for (int i = 0; i < params.size(); i++) {
                grads.push_back(grad_messages[params[i]->id]);
            }

            // Remove all nodes from the temporary constants
            temporary_constants.clear();

            // Restore the current group
            current_group = old_group;
            return grads;
        };

        // Copies the graph and optimizes it, populating the execution data
        Graph GraphInternal::optimize(const NodeVec &iTargets, 
            const Updates &iUpdates, 
            const NodeVec &iInputs,
            NodeVec &new_targets, 
            Updates &new_updates, 
            NodeVec &new_inputs) {

            logger()->debug() << "Running optimization of graph " << name;

            // Copy only the relevant part of the graph
            Graph optimize = create_graph();
            add_temporary_updates(iUpdates);

            NodeVec marked;
            marked.reserve(iTargets.size() + updates.size() + temporary_updates.size());
            for(auto& i : iTargets) marked.emplace_back(i);
            for(auto& u : updates) marked.emplace_back(u.second); // from member
            for(auto& t : temporary_updates) marked.emplace_back(t.second); // from member

            NodeVec mapping = this->copy(optimize.get(), get_ancestors_mask(marked));
            clear_temporary_updates();

            // Optimize
            for (size_t i = 0; i < optimize->nodes.size(); i++) {
                Node node = optimize->nodes[i];
                if (node->op->name == "Input") {
                    node->execution.inlined = true;
                }
                if (node->op->name == "Shared") {
                    node->execution.inlined = true;
                }
                if (node.is_scalar() and node.is_constant()) {
                    node->execution.inlined = true;
                }
                if (node->op->name == "Broadcast") {
                    node->execution.inlined = true;
                }
                if (node->op->name == "Transpose") {
                    node->execution.inlined = true;
                }
                if (node->op->name == "Neg") {
                    node->execution.inlined = true;
                }
                if (node->children.size() <= 1) {
                    node->execution.inlined = true;
                }
            }

            // Set the new_targets and new_updates
            for (int i = 0; i < iTargets.size(); i++) {
                new_targets.push_back(mapping[iTargets[i]->id]);
            }
            for (int i = 0; i < iUpdates.size(); i++) {
                Node node1 = mapping[iUpdates[i].first->id];
                Node node2 = mapping[iUpdates[i].second->id];
                new_updates.push_back(std::pair<Node, Node>(node1, node2));
            }
            for (int i = 0; i < iInputs.size(); i++) {
                new_inputs.push_back(mapping[iInputs[i]->id]);
            }
            return optimize;
        };

//        Node GraphInternal::shared_var(af::array value, std::string name) {
//            SharedPtr shared = std::make_shared<SharedVariable>(shared_vars.size(), value);
//            shared_vars.push_back(shared);
//            std::shared_ptr<Operator> op = std::make_shared<SharedInput>(this, shared);
//            Node node = derived_node(op);
//            node->name = name;
//            return node;
//        };
//        dType dtype;
//
//
//        if(value.type() == af::dtype::b8){
//            dtype = BOOLEAN;
//        } else if(value.type() == af::dtype::f32
//                  or value.type() == af::dtype::f64){
//            dtype = FLOAT;
//        } else {
//            dtype = UNSIGNED_INT;
//        }
//        af::dim4 dims = value.dims();
//
//        Node node = derived_node(op);
//
//        std::shared_ptr<NodeInternal> result = std::make_shared<NodeInternal>(
//                shared_from_this().get(),
//                default_device,
//                nodes.size(),
//                name,
//                ad_node_type::INPUT,
//                dtype,
//                Shape {dims[0], dims[1], dims[2], dims[3]},
//                std::make_shared<Input>(shared_from_this().get()),
//                0,
//                current_group
//        );
//        result->shared = std::make_shared<SharedVariable>(shared_vars.size(), value);
//        shared_vars.push_back(result->shared);
//        nodes.push_back(result);
//        result->op->owner = result;
//        return result;
//        };

        Node GraphInternal::derived_node(std::shared_ptr<Operator> op) {
            Node same_node = find_same_node(op);
            if (same_node.ptr.expired()) {
//            std::cout << "Creating new derived node for operator " << op->name << std::endl;
//                if(op->name == "Tanh"){
//                    std::cout << op->get_shape() << " " << op->get_parents()[0]->op->name <<
//                    op->get_parents()[0]->shape << std::endl;
//                }
                auto result = std::make_shared<NodeInternal>(
                        shared_from_this().get(),
                        default_device,
                        nodes.size(),
                        "Derived Node",
                        op->get_node_type(),
                        op->get_dtype(),
                        op->get_shape(),
                        op,
                        grad_level > op->get_grad_level() ? grad_level : op->get_grad_level(),
                        current_group
                );
                nodes.push_back(result);
                op->owner = result;
                NodeVec ancestors = op->get_ancestors();
                for (int i = 0; i < ancestors.size(); i++) {
                    ancestors[i]->children.push_back(result);
                }
                return result;
            } else {
                return same_node.alias();
            }
        }

        void GraphInternal::update_node(Node shared, Node update) {
            // Check the first node is a shared variable
            if (shared->op->name != "Shared") {
                auto err = InvalidArguments({shared, update}, "Update",
                                            "First argument can be only a SHARED_VARIABLE");
                logger()->error() << err.msg;
                throw err;
            }
            auto shared_shape = shared->shape;
            auto update_shape = update->shape;

            // Check that the shapes are correct
            for (int i = 0; i < 4; i++) {
                if (shared_shape[i] != update_shape[i]) {
                    auto err = IncompatibleShapes(NodeVec{shared, update}, "Update");
                    logger()->error() << err.msg;
                    throw err;
                }
            }

            // Check that the value types are correct
            if (shared->dtype != update->dtype) {
                auto err = InvalidArguments(NodeVec{shared, update}, "Update",
                                            "The Shared variable and the update should have the same dtype");
                logger()->error() << err.msg;
                throw err;
            }
            // Add it to the updates
            updates.push_back(std::pair<Node, Node>(shared, update));
        }

//    Node GraphInternal::constant_node(af::array value){
//        dType dtype;
//        if(value.type() == af::dtype::b8){
//            dtype = BOOLEAN;
//        } else if(value.type() == af::dtype::f32
//                  or value.type() == af::dtype::f64){
//            dtype = FLOAT;
//        } else {
//            dtype = INTEGER;
//        }
//        af::dim4 dims = value.dims();
//        std::shared_ptr<NodeInternal> result = std::make_shared<NodeInternal>(
//                shared_from_this().get(),
//                default_device,
//                nodes.size(),
//                "Constant Node",
//                ad_node_type::CONSTANT,
//                dtype,
//                Shape {dims[0], dims[1], dims[2], dims[3]},
//                std::make_shared<Input>(shared_from_this().get()),
//                0,
//                current_group
//        );
//        result->value = value;
//        nodes.push_back(result);
//        result->op->owner = result;
//        return result;
//    };

        SymInt GraphInternal::get_new_symbolic_integer() {
            this->sym_integer_count++;
            return SymInt::variable(this->sym_integer_count - 1);
        }

        Group GraphInternal::get_group(std::string full_name) {
            std::weak_ptr<NodeGroup> group = groups[0];
            std::stringstream name_stream(full_name);
            std::string item;
            while (std::getline(name_stream, item, GROUP_DELIMITER)) {
                size_t index = group.lock()->children.size();
                for (size_t i = 0; i < index; i++) {
                    if (item.compare(group.lock()->children[i].lock()->name) == 0) {
                        index = i;
                        break;
                    }
                }
                // If not found make a new group
                if (index == group.lock()->children.size()) {
                    groups.push_back(std::make_shared<NodeGroup>(item, group));
                    group.lock()->children.push_back(groups.back());
                    group = groups.back();
                } else {
                    group = group.lock()->children[index];
                }
            }
            return group;
        }

        void GraphInternal::set_group(std::string full_name) {
            current_group = get_group(full_name);
        }

        void GraphInternal::set_group(std::string name, Group parent) {
            current_group = get_group(parent.lock()->full_name + name);
        }

        void GraphInternal::reset_group() {
            current_group = groups[0];
        }

        Node GraphInternal::PI(){
            if(max_float == f64){
                return constant_value((double) M_PI);
            } else if(max_float == f32){
                return constant_value((float) M_PI);
            } else {
                auto err = OtherError(NodeVec{}, "'Pi' is not supported for max_float=" + to_string(max_float));
                logger()->error() << err.msg;
                throw err;
            }
        }

        Node GraphInternal::E(){
            if(max_float == f64){
                return constant_value((double) M_E);
            } else if(max_float == f32){
                return constant_value((float) M_E);
            } else {
                auto err = OtherError(NodeVec{}, "'e' is not supported for max_float=" + to_string(max_float));
                logger()->error() << err.msg;
                throw err;
            }
        }

        Node GraphInternal::LN_2(){
            if(max_float == f64){
                return constant_value((double) M_LN2);
            } else if(max_float == f32){
                return constant_value((float) M_LN2);
            } else {
                auto err = OtherError(NodeVec{}, "'Ln(2)' is not supported for max_float=" + to_string(max_float));
                logger()->error() << err.msg;
                throw err;
            }
        }

        Node GraphInternal::LN_10(){
            if(max_float == f64){
                return constant_value((double) M_LN10);
            } else if(max_float == f32){
                return constant_value((float) M_LN10);
            } else {
                auto err = OtherError(NodeVec{}, "'Ln(10)' is not supported for max_float=" + to_string(max_float));
                logger()->error() << err.msg;
                throw err;
            }
        }

//        Node GraphInternal::tensor4(dType dtype,
//                                    Shape shape,
//                                    std::string name) {
//            auto result = std::make_shared<NodeInternal>(
//                    shared_from_this().get(),
//                    default_device,
//                    nodes.size(),
//                    name,
//                    INPUT,
//                    dtype,
//                    shape,
//                    std::make_shared<op::Input>(shared_from_this().get(), dtype),
//                    0,
//                    current_group
//            );
//            nodes.push_back(result);
//            result->op->owner = result;
//            return result;
//        }

        Node GraphInternal::tensor4(dType dtype,
                                    SymInt shape0,
                                    SymInt shape1,
                                    SymInt shape2,
                                    SymInt shape3,
                                    std::string name) {
            Shape shape{shape0,
                                        shape1,
                                        shape2,
                                        shape3};
            return tensor4(dtype, shape, name);
        }

        Node GraphInternal::tensor4(dType dtype,
                                    std::string name) {
            Shape shape = {
                    get_new_symbolic_integer(),
                    get_new_symbolic_integer(),
                    get_new_symbolic_integer(),
                    get_new_symbolic_integer()
            };
            return tensor4(dtype, shape, name);
        }

        Node GraphInternal::tensor4_as(Node node, std::string name) {
            return tensor4(node->dtype, node->shape, name);
        }

        Node GraphInternal::tensor3(dType dtype,
                                    std::array<SymInt, 3> shape,
                                    std::string name) {
            return tensor4(dtype, {shape[0], shape[1], shape[2], SymInt::one}, name);
        }

        Node GraphInternal::tensor3(dType dtype,
                                    SymInt shape0,
                                    SymInt shape1,
                                    SymInt shape2,
                                    std::string name) {
            return tensor4(dtype, Shape{
                                   shape0,
                                   shape1,
                                   shape2,
                                   SymInt::one
                           },
                           name);
        }

        Node GraphInternal::tensor3(dType dtype,
                                    std::string name) {
            auto shape0 = get_new_symbolic_integer();
            auto shape1 = get_new_symbolic_integer();
            auto shape2 = get_new_symbolic_integer();
            return tensor3(dtype, shape0, shape1, shape2, name);
        }


        Node GraphInternal::tensor3_as(Node node, std::string name) {
            if (not node.is_tensor3()) {
                auto err = InvalidArguments(NodeVec{node}, "tensor3_as", "The variables is not a tensor3");
                logger()->error() << err.msg;
                throw err;
            }
            return tensor3(nodes[node->id]->dtype,
                           nodes[node->id]->shape[0],
                           nodes[node->id]->shape[1],
                           nodes[node->id]->shape[2],
                           name);
        }

        Node GraphInternal::matrix(dType dtype,
                                   std::array<SymInt, 2> shape,
                                   std::string name) {
            Shape shape_t{shape[0], shape[1], SymInt::one, SymInt::one};
            return tensor4(dtype, shape_t, name);
        }

        Node GraphInternal::matrix(dType dtype,
                                   SymInt shape0,
                                   SymInt shape1,
                                   std::string name) {
            return tensor4(dtype, Shape{
                                   shape0,
                                   shape1,
                                   SymInt::one,
                                   SymInt::one},
                           name);
        }

        Node GraphInternal::matrix(dType dtype,
                                   std::string name) {
            auto shape0 = get_new_symbolic_integer();
            auto shape1 = get_new_symbolic_integer();
            return matrix(dtype, shape0, shape1, name);
        }


        Node GraphInternal::matrix_as(Node node, std::string name) {
            if (not node.is_matrix()) {
                auto err = InvalidArguments(NodeVec{node}, "matrix_as", "The variables is not a matrix");
                logger()->error() << err.msg;
                throw err;
            }
            return matrix(nodes[node->id]->dtype,
                          nodes[node->id]->shape[0],
                          nodes[node->id]->shape[1],
                          name);
        }

        Node GraphInternal::square_matrix(dType dtype,
                                          SymInt shape,
                                          std::string name) {
            Shape shape_t{shape, shape, SymInt::one, SymInt::one};
            return tensor4(dtype, shape_t, name);
        }

        Node GraphInternal::vector(dType dtype,
                                   SymInt shape,
                                   std::string name) {
            Shape shape_t{shape, SymInt::one, SymInt::one, SymInt::one};
            return tensor4(dtype, shape_t, name);
        }

        Node GraphInternal::vector(dType dtype,
                                   std::string name) {
            auto shape0 = get_new_symbolic_integer();
            return vector(dtype, shape0, name);
        }


        Node GraphInternal::vector_as(Node node,
                                      std::string name) {
            if (not node.is_vector()) {
                auto err = InvalidArguments(NodeVec{node}, "vector_as", "The variables is not a vector");
                logger()->error() << err.msg;
                throw err;
            }
            return vector(nodes[node->id]->dtype,
                          nodes[node->id]->shape[0],
                          name);
        }

        Node GraphInternal::scalar(dType dtype,
                                   std::string name) {
            Shape shape_t{SymInt::one, SymInt::one, SymInt::one, SymInt::one};
            return tensor4(dtype, shape_t, name);
        }

        std::shared_ptr<const Operator> Operator::get_base_op(std::shared_ptr<const Operator> const op) {
            std::shared_ptr<const Operator> base_op = op;
            while (base_op->name == "Alias") {
                base_op = base_op->get_parents()[0]->op;
            }
            return base_op;
        }

        bool symbolic_equals(Node const & node1, Node const & node2) {
            if (node1->id == node2->id) {
                return true;
            }
            if (node1->node_type != node2->node_type) {
                return false;
            }
            nodeType node_type = node1->node_type;
            if (node_type == INPUT) {
                return false;
            } else {
                auto base_op1 = Operator::get_base_op(node1->op);
                auto base_op2 = Operator::get_base_op(node2->op);
                return base_op1->equals(base_op2) or base_op2->equals(base_op1);
            }
        };
    }
}
#endif //METADIFF_CORE_IMPL_H
