//
// Created by alex on 10/12/15.
//

#ifndef METADIFF_VISUALIZATION_DAGRE_H
#define METADIFF_VISUALIZATION_DAGRE_H

namespace metadiff {
    namespace dagre {
        /**
         * Writes the template part of the html at the top
         */
        void print_html_headers(std::ofstream& f, Graph& graph);
        /**
         * Writes the template part of the html at the bottom
         */
        void print_html_footers(std::ofstream& f, Graph& graph);
        /**
         * Generates the javascript for the node
         */
        void print_node(std::ofstream& f, Node node);
        /**
         * Generates the javascript for all edges going in to the node
         */
        void print_edges(std::ofstream& f, Node node);
        /**
         * Generates the javascript for the node representing this update
         */
        void print_update_node(std::ofstream& f, std::pair<Node,Node> update);
        /**
         * Generates the javascript for the edge representing this update
         */
        void print_update_edge(std::ofstream& f, std::pair<Node,Node> update);

        void print_to_file(std::string file_path,
                           Graph graph,
                           Updates& updates) {
            // Open file
            std::ofstream f;
            f.open(file_path);

            // Print headers
            print_html_headers(f, graph);
            f << "\t// Create a new graph morpher\n"
                    "\tvar morpher = new GraphMorpher(\"g\", 1000, 600);\n";

            // Print all groups
            f << "\t// Set all groups\n";
            for(size_t i=1;i<graph->groups.size(); i++){
                f << "\tmorpher.addGroup(\"_root/"
                << graph->groups[i]->get_full_name()
                << "\");\n";
            }

            // Print all nodes
            f << "\n\t// Add all nodes\n";
            for(size_t i=0; i < graph->nodes.size(); i++){
                print_node(f, graph->nodes[i]);
            }

            // Print graph updates
            f << "\n\t// Add graph update nodes\n";
            for (size_t i=0; i<graph->updates.size(); i++){
                print_update_node(f, graph->updates[i]);
            }

            // Print temporary updates
            f << "\n\t// Add temporary update nodes\n";
            for (size_t i=0; i<updates.size(); i++){
                print_update_node(f, updates[i]);
            }

            // Print all edges
            f << "\n\t// Add all edges\n";
            for(size_t i=0; i < graph->nodes.size(); i++){
                print_edges(f, graph->nodes[i]);
            }

            // Print all graph update edges
            f << "\n\t// Add graph update edges\n";
            for(size_t i=0; i<graph->updates.size(); i++){
                print_update_edge(f, graph->updates[i]);
            }

            // Print all temporary update edges
            f << "\n\t// Add temporary update edges\n";
            for(size_t i=0; i<updates.size(); i++){
                print_update_edge(f, updates[i]);
            }

            // Code to render the graph
            f << "\t// Render graph\n\tmorpher.populateSVG();\n";

            // Print footers
            print_html_footers(f, graph);

            // Close file
            f.close();
        }

        void print_to_file(std::string file_path, Graph graph){
            Updates updates;
            print_to_file(file_path, graph, updates);
        }

        /**
         * Helper function to print the name of a node
         */
        std::ofstream& print_name(std::ofstream& f, Node node_in) {
            auto node = node_in.unwrap();
            if(node-> type == ad_node_type::SYMBOLIC_INTEGER){
                f << "SYMINT[" << node->id << "]";
            } else if (node->type == ad_node_type::CONSTANT) {
                if(node_in.is_scalar() and node->op->name == "Value"){
                    std::shared_ptr<ConstantValue> cast_op = std::static_pointer_cast<ConstantValue>(node->op);
                    f << cast_op->value << "[" << node->id << "]";
                } else if(node->op->name != "Input") {
                    f <<  node->op->name << "[" << node->id << "]";
                } else {
                    f << "CONST[" << node->id << "]";
                }
            } else if (node->type == ad_node_type::INPUT or
                       node->type == ad_node_type::SHARED_INPUT) {
                f << node->name << "[" << node->id << "]";
            } else {
                f << node->op->name << "[" << node->id << "]";
            }
            return f;
        }

        /**
         * Helper function to print the correct color of the node
         */
        std::ofstream& print_color(std::ofstream& f, Node node){
            switch(node.unwrap()->type){
                case INPUT: f << "#00ff00"; return f;
                case SHARED_INPUT:  f << "#006400"; return f;
                case INPUT_DERIVED: f << "#0000ff"; return f;
                case CONSTANT: f << "#ffff00"; return f;
                case CONSTANT_DERIVED: f << "#ffa500"; return f;
                default: f << "#800080"; return f;
            }
        }

        /**
         * Helper function to print the correct shape of the node
         */
        std::ofstream& print_shape(std::ofstream& f, Node node){
            switch(node.unwrap()->type){
                case INPUT: f << "rect"; return f;
                case SHARED_INPUT:  f << "rect"; return f;
                case INPUT_DERIVED: f << "ellipse"; return f;
                case CONSTANT: f << "circle"; return f;
                case CONSTANT_DERIVED: f << "ellipse"; return f;
                default: f << "rect"; return f;
            }
        }

        /**
         * Helper function to print a list of the ids of the vector of nodes
         */
        std::ofstream& print_ids(std::ofstream&f, NodeVec nodes){
            if(nodes.size() == 0){
                f << "()";
            } else if(nodes.size() == 1){
                f << "(" << nodes[0].unwrap()->id << ")";
            } else {
                f << "(";
                for (size_t i = 0; i < nodes.size() - 1; i++) {
                    f << nodes[i].unwrap()->id << ", ";
                }
                f << nodes[nodes.size()-1].unwrap()->id << ")";
            }
            return f;
        }

        /**
         * Helper function to print the description of a node
         */
        std::ofstream& print_description(std::ofstream&f, Node node_in){
            auto node = node_in.unwrap();
            f << "\t\tdescription: \n";
            f << "\t\t\t\"Name: " << node->name << " <br>\"+\n";
            f << "\t\t\t\"Group: " << node->group.lock()->name << " <br>\"+\n";
            f << "\t\t\t\"Type: " << node->type << " <br>\"+\n";
            f << "\t\t\t\"Value type: " << node->v_type << " <br>\"+\n";
            f << "\t\t\t\"Shape: (" <<
            node->shape[0] << ", " <<
            node->shape[1] << ", " <<
            node->shape[2] << ", " <<
            node->shape[3] << ") <br>\"+\n";
            f << "\t\t\t\"Device: " << node->device << " <br>\"+\n";
            f << "\t\t\t\"Gradient Level:" << node->grad_level << " <br>\"+\n";
            f << "\t\t\t\"Parents: ";
            print_ids(f , node->op->get_ancestors()) << " <br>\"+\n";
            f << "\t\t\t\"Children: ";
            print_ids(f , node->children) << " <br>\"\n";
            return f;
        }

        /**
         * Helper function to return true if the node is constant input
         */
        bool is_constant(Node node){
            if(node.unwrap()-> type == CONSTANT or node.unwrap()-> type == SYMBOLIC_INTEGER){
                std::string op_name = node.unwrap()->op->name;
                return op_name == "Input" or op_name == "Value" or op_name == "Zeros" or op_name == "Ones";
            }
            return false;
        }

        void print_node(std::ofstream& f, Node node){
            // The javascript code is:
            // moprpher.addNode("<group>", "<node name>", "{<node attributes>}");
            if(is_constant(node)){
                // For constants we create a node for each single connection
                for(size_t i=0;i<node.unwrap()->children.size(); i++){
                    f << "\tmorpher.addNode(\"_root/" << node.unwrap()->group.lock()->get_full_name() <<
                    "\",\n\t\t\"CONST[" << node.unwrap()->id << "]_" << i << "\", {\n";
                    f << "\t\tlabel: \"\",\n";
                    f << "\t\tshape: \"";
                    print_shape(f, node) << "\",\n";
                    // Print the description
                    print_description(f, node) << "});\n";
                }
            } else {
                f << "\tmorpher.addNode(\"_root/" << node.unwrap()->group.lock()->get_full_name() <<
                "\",\n\t\t\"";
                print_name(f, node) << "\", {\n";
                f << "\t\tlabel: \"";
                print_name(f, node) << "\",\n";
                f << "\t\tshape: \"";
                print_shape(f, node) << "\",\n";
                // Print the description
                print_description(f, node) << "});\n";
            }
        }

        void print_edges(std::ofstream& f, Node node){
            // The javascript code is:
            // moprpher.addEdge("<ancestor name>", "<node name>", "<label>");
            NodeVec ancestors = node.unwrap()->op->get_ancestors();
            for (size_t i = 0; i < ancestors.size(); i++) {
                if(is_constant(ancestors[i])){
                    // If parent is constant have to connect correctly
                    for(size_t ind=0; ind<ancestors[i].unwrap()->children.size();ind ++){
                        if(ancestors[i].unwrap()->children[ind].unwrap()->id == node.unwrap()->id){
                            f << "\tmorpher.addEdge(\"CONST[" << ancestors[i].unwrap()->id << "]_" << ind << "\", \"";
                            print_name(f, node) << "\", \"" << i << "\");\n";
                        }
                    }
                } else {
                    f << "\tmorpher.addEdge(\"";
                    print_name(f, ancestors[i]) << "\", \"";
                    print_name(f, node) << "\", \"" << i << "\");\n";
                }
            }
        }

        void print_update_node(std::ofstream& f, std::pair<Node,Node> update){
            // The javascript code is:
            // moprpher.addNode("<group>", "<node name>", "{<node attributes>}");
            auto first = update.first.unwrap();
            auto second = update.second.unwrap();
            f << "\tmorpher.addNode(\"_root/" << second->group.lock()->get_full_name() << "\", "
            << "\"Update " <<  first->name << "[" << first->id << "]\", {\n"
            << "\t\tstyle: \"fill: #FF1493\",\n"
            << "\t\tdescription: \"\"\n"
            << "\t});\n";
        }

        void print_update_edge(std::ofstream& f, std::pair<Node,Node> update){
            // The javascript code is:
            // moprpher.addEdge("<ancestor name>", "<node name>", "<label>");
            auto first = update.first.unwrap();
            f << "\tmorpher.addEdge(\"";
            print_name(f, update.second) << "\", \"Update "
            << first->name << "[" << first->id << "]\", \"\");\n";
        }

        void print_html_headers(std::ofstream& f, Graph& graph){
            f << ""
                    "<!DOCTYPE html>\n"
                    "<!-- saved from url=(0065)http://cpettitt.github.io/project/dagre-d3/latest/demo/hover.html -->\n"
                    "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"><meta charset=\"utf-8\">\n"
                    "<title>Function Graph</title>\n"
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
                    "<h5 style=\"margin-top: 0px; margin-bottom: 0px\">Broadcast policy: " << graph->broadcast << "</h5>\n\n"
                    "<svg width=\"1500\" height=\"900\"></svg>\n"
                    "\n"
                    "<script id=\"js\">\n";
        }

        void print_html_footers(std::ofstream& f, Graph& graph){
            f <<"</script>\n</body></html>\n";
        }
    }
}

#endif //METADIFF_VISUALIZATION_DAGRE_H
