//
// Created by alex on 10/12/15.
//

#ifndef METADIFF_VISUALIZATION_DAGRE_H
#define METADIFF_VISUALIZATION_DAGRE_H

namespace metadiff {
    namespace dagre {
        bool exists_in(Node node, std::vector<Node>& nodes){
            for(int i=0;i<nodes.size();i++){
                if(nodes[i].ptr == node.ptr){
                    return true;
                }
            }
            return false;
        }

        std::string node_name_html(Node node) {
            if(node.ptr->type == ad_node_type::SYMBOLIC_INTEGER){
                return "SYMINT[" + std::to_string(node.ptr->id) + "]";
            } else if (node.ptr->type == ad_node_type::CONSTANT) {
                if(node.is_scalar()){
//                    std::string value;
//                    if(node.ptr->v_type == FLOAT) {
//                        float host[1];
//                        node.ptr->value.host(host);
//                        value = std::to_string(host[0]);
//                    } else if(node.ptr->v_type == INTEGER){
//                        int host[1];
//                        node.ptr->value.host(host);
//                        value = std::to_string(host[0]);
//                    } else {
//                        bool host[1];
//                        node.ptr->value.host(host);
//                        value = std::to_string(host[0]);
//                    }
                    std::stringstream name;
                    name << node.ptr->op->get_scalar_value() << "[" << node.ptr->id << "]";
                    return name.str();
                } else if(node.ptr->op->name != "Input") {
                    return node.ptr->op->name + "[" + std::to_string(node.ptr->id) + "]";
                } else {
                    return "CONST[" + std::to_string(node.ptr->id) + "]";
                }
            } else if (node.ptr->type == ad_node_type::INPUT) {
                return node.ptr->name +  "[" + std::to_string(node.ptr->id) + "]";
            } else if (node.ptr->type == ad_node_type::SHARED_INPUT) {
                return node.ptr->name + "[" + std::to_string(node.ptr->id) + "]";
            } else if (node.ptr->type == ad_node_type::UPDATE) {
                return "Update[" + std::to_string(node.ptr->id) + "]";
            } else {
                return node.ptr->op->name + "[" + std::to_string(node.ptr->id) + "]";
            }
        }

        std::string node_color_html(Node node, std::vector<Node> targets) {
            if (exists_in(node, targets)) {
                return "#ff0000";
            } else {
                switch(node.ptr->type){
                    case INPUT: return "#00ff00";
                    case SHARED_INPUT:  return "#006400";
                    case INPUT_DERIVED: return "#0000ff";
                    case CONSTANT: return "#ffff00";
                    case CONSTANT_DERIVED: return "#ffa500";
                    case UPDATE: return "#FF1493";
                    default: return "#800080";
                }
            }
        }

        void node_to_html(Node node, std::vector<Node> &targets,
                          std::vector<std::string> &names) {
            std::string state_name = node_name_html(node);
            std::string state_color = node_color_html(node, targets);

            NodeVec ancestors = node.ptr->op->get_ancestors();
            std::string anc_id_str = "[";
            for (int i = 0; i < ancestors.size(); i++) {
                anc_id_str += std::to_string(ancestors[i].ptr->id);
                if (i < ancestors.size() - 1) {
                    anc_id_str += ", ";
                }
            }
            anc_id_str += "]";
            std::string child_id_str = "[";
            for (int i=0;i<node.ptr->children.size(); i++){
                child_id_str += std::to_string(node.ptr->children[i].ptr->id);
                if (i < node.ptr->children.size() - 1) {
                    child_id_str += ", ";
                }
            }
            child_id_str += "]";
            names.push_back("'" + state_name + "': {\n"
                    "\t\tdescription: \"Name: " + node.ptr->name + " <br> "
                                    "Type: " + to_string(node.ptr->type) + " <br> "
                                    "Device: " + to_string(node.ptr->device) + " <br> "
                                    "Value type: " + to_string(node.ptr->v_type) + " <br> "
                                    "Shape: [" + node.ptr->shape[0].to_string() + ", " +
                            node.ptr->shape[1].to_string() + ", " + node.ptr->shape[2].to_string() + ", " +
                            node.ptr->shape[3].to_string() + "] <br> "
                                    "Gradient Level:" + std::to_string(node.ptr->grad_level) + " <br> "
                                    "Parents: " + anc_id_str + " <br> "
                                    "Children: " + child_id_str +
                            "\",\n\t\tstyle: \"fill: " + state_color + "\"\n"
                                    "\t}");
        }

        void edges_to_html(Node node, std::vector<std::string> &edges) {
            std::string state_name = node_name_html(node);
            auto ancestors = node.ptr->op->get_ancestors();
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
                           const Updates& updates) {
            graph->add_temporary_updates(updates);

            size_t max_grad_level = 0;
            for (size_t i = 0; i < targets.size(); i++) {
                if (targets[i].ptr->grad_level > max_grad_level) {
                    max_grad_level = targets[i].ptr->grad_level;
                }
            }
            std::vector<std::string> nodes;
            std::vector<std::string> edges;
            for (int i = 0; i < graph->nodes.size(); i++) {
                node_to_html(graph->nodes[i], targets, nodes);
                edges_to_html(graph->nodes[i], edges);
            }
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
                    "// Create a new directed graph\n"
                    "var g = new dagreD3.graphlib.Graph({compound:true, multigraph: true})\n"
                    "\t\t.setGraph({})\n"
                    "\t\t.setDefaultEdgeLabel(function() { return {}; });\n";
            // Print the nodes
            f << "// Nodes states\n"
                    "var states = {\n";
            for (int i = 0; i < nodes.size(); i++) {
                if (i == nodes.size() - 1) {
                    f << "\t" << nodes[i] << "\n";
                } else {
                    f << "\t" << nodes[i] << ",\n";
                }
            }
            f << "}\n";
            // Print grad boxes
//            f << "// Set the gradient level boxes\n";
//            for (int i = 0; i <= max_grad_level; i++) {
//                auto name = "Calculation Level[" + std::to_string(i) + "]";
//                f << "g.setNode('grad_" + std::to_string(i) + "', {label: '" + name +
//                     "', clusterLabelPos: 'top', style: 'fill: #d3d7e8'});\n";
//            }

            f << "// Add states to the graph, set labels, and style\n"
                    "Object.keys(states).forEach(function(state) {\n"
                    "  var value = states[state];\n"
                    "  value.label = state;\n"
                    "  value.rx = value.ry = 5;\n"
                    "  g.setNode(state, value);\n"
                    "});\n";

            // Print the edge connections
            f << "// Set up the edges\n";
            for (int i = 0; i < edges.size(); i++) {
                f << edges[i] << "\n";
            }
            f << "\n";

            f << "// Create the renderer\n"
                    "var render = new dagreD3.render();\n"
                    "\n"
                    "// Set up an SVG group so that we can translate the final graph.\n"
                    "var svg = d3.select(\"svg\"),\n"
                    "    inner = svg.append(\"g\");\n"
                    "\n"
                    "// Set up zoom support\n"
                    "var zoom = d3.behavior.zoom().on(\"zoom\", function() {\n"
                    "    inner.attr(\"transform\", \"translate(\" + d3.event.translate + \")\" +\n"
                    "                                \"scale(\" + d3.event.scale + \")\");\n"
                    "  });\n"
                    "svg.call(zoom);\n"
                    "\n"
                    "// Simple function to style the tooltip for the given node.\n"
                    "var styleTooltip = function(name, description) {\n"
                    "  return \"<p class='name'>\" + name + \"</p>"
                    "<p class='description'  style=\\\"text-align:left\\\">\" + description + \"</p>\";\n"
                    "};\n"
                    "\n"
                    "// Run the renderer. This is what draws the final graph.\n"
                    "render(inner, g);\n"
                    "\n"
                    "inner.selectAll(\"g.node\")\n"
                    "  .attr(\"title\", function(v) { return styleTooltip(v, g.node(v).description) })\n"
                    "  .each(function(v) { $(this).tipsy({ gravity: \"w\", opacity: 1, html: true }); });\n"
                    "\n"
                    "// Center the graph\n"
                    "var initialScale = 0.75;\n"
                    "zoom\n"
                    "  .translate([(svg.attr(\"width\") - g.graph().width * initialScale) / 2, 20])\n"
                    "  .scale(initialScale)\n"
                    "  .event(svg);\n"
                    "svg.attr('height', g.graph().height * initialScale + 40);\n"
                    "</script>\n"
                    "</body></html>\n";
            f.close();
            graph->clear_temporary_updates();
        }
    }
}

#endif //METADIFF_VISUALIZATION_DAGRE_H
