//
// Created by alex on 10/12/15.
//

#ifndef METADIFF_VISUALIZATION_DAGRE_H
#define METADIFF_VISUALIZATION_DAGRE_H

namespace metadiff {
    namespace dagre {
        std::string node_name_html(Node node) {
            if(node.unwrap()->type == ad_node_type::SYMBOLIC_INTEGER){
                return "SYMINT[" + std::to_string(node.unwrap()->id) + "]";
            } else if (node.unwrap()->type == ad_node_type::CONSTANT) {
                if(node.is_scalar() and node.unwrap()->op->name == "Value"){
//                    std::string value;
//                    if(node.unwrap()->v_type == FLOAT) {
//                        float host[1];
//                        node.unwrap()->value.host(host);
//                        value = std::to_string(host[0]);
//                    } else if(node.unwrap()->v_type == INTEGER){
//                        int host[1];
//                        node.unwrap()->value.host(host);
//                        value = std::to_string(host[0]);
//                    } else {
//                        bool host[1];
//                        node.unwrap()->value.host(host);
//                        value = std::to_string(host[0]);
//                    }
                    std::shared_ptr<ConstantValue> cast_op = std::static_pointer_cast<ConstantValue>(node.unwrap()->op);
                    std::stringstream name;
                    name << cast_op->value << "[" << node.unwrap()->id << "]";
                    return name.str();
                } else if(node.unwrap()->op->name != "Input") {
                    return node.unwrap()->op->name + "[" + std::to_string(node.unwrap()->id) + "]";
                } else {
                    return "CONST[" + std::to_string(node.unwrap()->id) + "]";
                }
            } else if (node.unwrap()->type == ad_node_type::INPUT) {
                return node.unwrap()->name +  "[" + std::to_string(node.unwrap()->id) + "]";
            } else if (node.unwrap()->type == ad_node_type::SHARED_INPUT) {
                return node.unwrap()->name + "[" + std::to_string(node.unwrap()->id) + "]";
//            } else if (node.unwrap()->type == ad_node_type::UPDATE) {
//                return "Update[" + std::to_string(node.unwrap()->id) + "]";
            } else {
                return node.unwrap()->op->name + "[" + std::to_string(node.unwrap()->id) + "]";
            }
        }

        std::string node_color_html(Node node, std::vector<Node> targets) {
            // Check if it is a target
            for(int i=0;i<targets.size();i++){
                if(targets[i].unwrap() == node.unwrap()){
                    return "#ff0000";
                }
            }
            // Check if it is an update
//            Updates updates = node.unwrap()->graph->updates;
//            for(size_t i=0;i<updates.size(); i++){
//                if(node.unwrap() == updates[i].second.unwrap()){
//                    return "#FF1493";
//                }
//            }
            switch(node.unwrap()->type){
                case INPUT: return "#00ff00";
                case SHARED_INPUT:  return "#006400";
                case INPUT_DERIVED: return "#0000ff";
                case CONSTANT: return "#ffff00";
                case CONSTANT_DERIVED: return "#ffa500";
                default: return "#800080";
            }
        }

        void write_node(Node node, std::vector<Node> &targets, std::ofstream& f ){
            f << "\tmorpher.addNode(\"_root/"
            << node.unwrap()->group.lock()->get_full_name() << "\", ";
            std::string state_name = node_name_html(node);
            std::string state_color = node_color_html(node, targets);
            f << "\"" << state_name << "\", {\n";

            NodeVec ancestors = node.unwrap()->op->get_ancestors();
            std::string anc_id_str = "[";
            for (int i = 0; i < ancestors.size(); i++) {
                anc_id_str += std::to_string(ancestors[i].unwrap()->id);
                if (i < ancestors.size() - 1) {
                    anc_id_str += ", ";
                }
            }
            anc_id_str += "]";
            std::string child_id_str = "[";
            for (int i=0;i<node.unwrap()->children.size(); i++){
                child_id_str += std::to_string(node.unwrap()->children[i].unwrap()->id);
                if (i < node.unwrap()->children.size() - 1) {
                    child_id_str += ", ";
                }
            }
            child_id_str += "]";
            f << "\t\tdescription: \"Name: " + node.unwrap()->name + " <br> "
                    "Type: " + to_string(node.unwrap()->type) + " <br> "
                         "Device: " + to_string(node.unwrap()->device) + " <br> "
                         "Value type: " + to_string(node.unwrap()->v_type) + " <br> "
                         "Shape: [" + node.unwrap()->shape[0].to_string() + ", " +
                 node.unwrap()->shape[1].to_string() + ", " + node.unwrap()->shape[2].to_string() + ", " +
                 node.unwrap()->shape[3].to_string() + "] <br> "
                         "Gradient Level:" + std::to_string(node.unwrap()->grad_level) + " <br> "
                         "Parents: " + anc_id_str + " <br> "
                         "Children: " + child_id_str + "<br>"
                         "Group: " + node.unwrap()->group.lock()->name +
                 "\",\n\t\tstyle: \"fill: " + state_color + "\",\n"
                    "\t\tlabel: \"" << state_name << "\"});\n";

        }

        void write_node_edges(Node node, std::vector<Node> &targets, std::ofstream& f) {
            std::string state_name = node_name_html(node);
            auto ancestors = node.unwrap()->op->get_ancestors();
            std::vector<std::string> anc_names;
            for (int i = 0; i < ancestors.size(); i++) {
                anc_names.push_back(node_name_html(ancestors[i]));
            }

            for (int i = 0; i < ancestors.size(); i++) {
                f << "\tmorpher.addEdge(\"" << anc_names[i]
                << "\", \"" << state_name << "\", \""
                << i << "\");\n";
            }
        }

        void node_to_html(Node node, std::vector<Node> &targets,
                          std::vector<std::string> &names) {
            std::string state_name = node_name_html(node);
            std::string state_color = node_color_html(node, targets);

            NodeVec ancestors = node.unwrap()->op->get_ancestors();
            std::string anc_id_str = "[";
            for (int i = 0; i < ancestors.size(); i++) {
                anc_id_str += std::to_string(ancestors[i].unwrap()->id);
                if (i < ancestors.size() - 1) {
                    anc_id_str += ", ";
                }
            }
            anc_id_str += "]";
            std::string child_id_str = "[";
            for (int i=0;i<node.unwrap()->children.size(); i++){
                child_id_str += std::to_string(node.unwrap()->children[i].unwrap()->id);
                if (i < node.unwrap()->children.size() - 1) {
                    child_id_str += ", ";
                }
            }
            child_id_str += "]";
            names.push_back("'" + state_name + "': {\n"
                    "\t\tdescription: \"Name: " + node.unwrap()->name + " <br> "
                                    "Type: " + to_string(node.unwrap()->type) + " <br> "
                                    "Device: " + to_string(node.unwrap()->device) + " <br> "
                                    "Value type: " + to_string(node.unwrap()->v_type) + " <br> "
                                    "Shape: [" + node.unwrap()->shape[0].to_string() + ", " +
                            node.unwrap()->shape[1].to_string() + ", " + node.unwrap()->shape[2].to_string() + ", " +
                            node.unwrap()->shape[3].to_string() + "] <br> "
                                    "Gradient Level:" + std::to_string(node.unwrap()->grad_level) + " <br> "
                                    "Parents: " + anc_id_str + " <br> "
                                    "Children: " + child_id_str + "<br>"
                                    "Group: " + node.unwrap()->group.lock()->name +
                            "\",\n\t\tstyle: \"fill: " + state_color + "\"\n"
                                    "\t}");
        }

        void edges_to_html(Node node, std::vector<std::string> &edges) {
            std::string state_name = node_name_html(node);
            auto ancestors = node.unwrap()->op->get_ancestors();
            std::vector<std::string> anc_names;
            for (int i = 0; i < ancestors.size(); i++) {
                anc_names.push_back(node_name_html(ancestors[i]));
            }

            for (int i = 0; i < ancestors.size(); i++) {
                edges.push_back(
                        "g.setEdge({v:'" + anc_names[i] + "', w:'" + state_name + "', name:'" + std::to_string(i) +
                        "'}, {label: \"" + std::to_string(i) + "\"});");
            }
//            edges.push_back("g.setParent('" + state_name + "', 'grad_" + std::to_string(node->grad_level) + "');");
        }

        void dagre_to_file(std::string file_name, Graph graph,
                           std::vector<Node> targets,
                           Updates& nupdates) {
            graph->add_temporary_updates(nupdates);

            size_t max_grad_level = 0;
            for (size_t i = 0; i < targets.size(); i++) {
                if (targets[i].unwrap()->grad_level > max_grad_level) {
                    max_grad_level = targets[i].unwrap()->grad_level;
                }
            }
//            std::vector<std::string> nodes;
//            std::vector<std::string> edges;
//            for (int i = 0; i < graph->nodes.size(); i++) {
//                node_to_html(graph->nodes[i], targets, nodes);
//                edges_to_html(graph->nodes[i], edges);
//            }
//            for (size_t i=0; i<nupdates.size(); i++){
//                std::pair<Node, Node> update = nupdates[i];
//                std::string update_name = "Update " + update.first.unwrap()->name + "[" +
//                                          std::to_string(update.first.unwrap()->id) + "]";
//                nodes.push_back("'" + update_name +  "': {\n"
//                        "\t\tdescription: \"\",\n"
//                        "\t\tstyle: \"fill: #FF1493\"\n"
//                        "\t}\n");
//                edges.push_back(
//                        "g.setEdge({v:'" + node_name_html(update.second) + "', w:'" +
//                        update_name + "', name:'Update'}, {label: \"Update\"});");
//            }
            std::ofstream f;
            f.open(file_name);
            f << ""
                    "<!DOCTYPE html>\n"
                    "<!-- saved from url=(0065)http://cpettitt.github.io/project/dagre-d3/latest/demo/hover.html -->\n"
                    "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"><meta charset=\"utf-8\">\n"
            << "<title>Function Graph</title>\n"
                    "\n"
                    "<link rel=\"stylesheet\" href=\"../../html_files/demo.css\">\n"
                    "<script src=\"../../html_files/d3.v3.min.js\" charset=\"utf-8\"></script>\n"
                    "<script src=\"../../html_files/dagre-d3.js\"></script><style type=\"text/css\"></style>\n"
                    "\n"
                    "<!-- Pull in JQuery dependencies -->\n"
                    "<link rel=\"stylesheet\" href=\"../../html_files/tipsy.css\">\n"
                    "<script src=\"../../html_files/jquery-1.9.1.min.js\"></script>\n"
                    "<script src=\"../../html_files/tipsy.js\"></script>\n"
                    "<script src=\"../../html_files/graph.js\"></script>\n"
                    "\n"
                    "</head><body style=\"margin:0;padding:0\" height=\"100%\">"
                    "<h3 style=\"margin-bottom: 0px\">" << graph->name << "</h3>\n"
                    "<h5 style=\"margin-top: 0px; margin-bottom: 0px\">Default device: " << graph->default_device << "</h5>\n"
                    "<h5 style=\"margin-top: 0px; margin-bottom: 0px\">Float type: " << graph->f_type << "</h5>\n"
                    "<h5 style=\"margin-top: 0px; margin-bottom: 0px\">Integer type: " << graph->i_type << "</h5>\n"
                    "<h5 style=\"margin-top: 0px; margin-bottom: 0px\">Broadcast policy: " << graph->broadcast << "</h5>\n"
                    "\n"
                    "<style id=\"css\">\n"
                    "text {\n"
                    "  font-weight: 300;\n"
                    "  font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serf;\n"
                    "  font-size: 14px;\n"
                    "}\n"
                    "\n"
                    ".node rect {\n"
                    "  stroke: #333;\n"
                    "  fill: #fff;\n"
                    "}\n"
                    "\n"
                    ".edgePath path {\n"
                    "  stroke: #333;\n"
                    "  fill: #333;\n"
                    "  stroke-width: 1.5px;\n"
                    "}\n"
                    "\n"
                    ".node text {\n"
                    "  pointer-events: none;\n"
                    "}\n"
                    "\n"
                    "/* This styles the title of the tooltip */\n"
                    ".tipsy .name {\n"
                    "  font-size: 1.5em;\n"
                    "  font-weight: bold;\n"
                    "  color: #60b1fc;\n"
                    "  margin: 0;\n"
                    "}\n"
                    "\n"
                    "/* This styles the body of the tooltip */\n"
                    ".tipsy .description {\n"
                    "  font-size: 1.2em;\n"
                    "}\n"
                    "</style>\n"
                    "\n"
                    "<svg width=\"1500\" height=\"900\"></svg>\n"
                    "\n"
                    "<script id=\"js\">\n"
                    "\t// Create a new graph morpher\n"
                    "\tvar morpher = new GraphMorpher(\"g\", 1000, 600);\n";
            // Print groups
            f << "\t// Set all groups\n";
            for(size_t i=1;i<graph->groups.size(); i++){
                f << "\tmorpher.addGroup(\"_root/"
                << graph->groups[i]->get_full_name()
                << "\");\n";
            }

            // Print the nodes
            f << "\n\t// Add all nodes\n";
            for(size_t i=0; i < graph->nodes.size(); i++){
                write_node(graph->nodes[i], targets, f);
            }

            // Print updates
            f << "\n\t// Add all update nodes\n";
            for (size_t i=0; i<nupdates.size(); i++){
                std::pair<Node, Node> update = nupdates[i];
                std::string update_name = "Update " + update.first.unwrap()->name + "[" +
                                          std::to_string(update.first.unwrap()->id) + "]";

                f << "\t\tmorpher.addNode(\"_root/"
                << update.second.unwrap()->group.lock()->get_full_name()
                << "\", \"" << update_name << "\", {\n"
                        "\t\tdescription: \"\",\n"
                        "\t\tstyle: \"fill: #FF1493\"\n"
                        "\t});\n";
            }

            // Print the edges
            f << "\n\t// Add all edges\n";
            for(size_t i=0; i < graph->nodes.size(); i++){
                write_node_edges(graph->nodes[i], targets, f);
            }

            f << "\n\t// Add all update edges\n";
            for(size_t i=0; i<nupdates.size(); i++){
                std::string update_name = "Update " + nupdates[i].first.unwrap()->name + "[" +
                                          std::to_string(nupdates[i].first.unwrap()->id) + "]";
                f << "\tmorpher.addEdge(\"" << node_name_html(nupdates[i].second)
                << "\", \"" << update_name << "\", \"\");\n";
            }
//            for (int i = 0; i < nodes.size(); i++) {
//                if (i == nodes.size() - 1) {
//                    f << "\t" << nodes[i] << "\n";
//                } else {
//                    f << "\t" << nodes[i] << ",\n";
//                }
//            }
//            f << "}\n";
            // Print grad boxes
//            f << "// Set the gradient level boxes\n";
//            for (int i = 0; i <= max_grad_level; i++) {
//                auto name = "Calculation Level[" + std::to_string(i) + "]";
//                f << "g.setNode('grad_" + std::to_string(i) + "', {label: '" + name +
//                     "', clusterLabelPos: 'top', style: 'fill: #d3d7e8'});\n";
//            }

//            f << "// Add states to the graph, set labels, and style\n"
//                    "Object.keys(states).forEach(function(state) {\n"
//                    "  var value = states[state];\n"
//                    "  value.label = state;\n"
//                    "  value.rx = value.ry = 5;\n"
//                    "  g.setNode(state, value);\n"
//                    "});\n";

            // Print the edge connections
//            f << "// Set up the edges\n";
//            for (int i = 0; i < edges.size(); i++) {
//                f << edges[i] << "\n";
//            }
//            f << "\n";

            f << "\t// Render graph\n"
                    "\tmorpher.populateSVG();\n"
                    "</script>\n"
                    "</body></html>\n";
            f.close();
            graph->clear_temporary_updates();
        }
    }
}

#endif //METADIFF_VISUALIZATION_DAGRE_H
