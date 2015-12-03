//
// Created by alex on 19/03/15.
//

#ifndef _AUTODIFF_AUTODIFF_H_
#define _AUTODIFF_AUTODIFF_H_

#include "vector"
#include "iostream"
#include <fstream>
#include "memory"
#include <unordered_map>
#include "arrayfire.h"

namespace diff {

    typedef size_t NodeId;
    enum TYPE {CONST, INPUT, INPUT_DERIVED};

    class Graph;
    class Node;


    class Operator{
    public:
        std::string name;
        Operator(std::string name): name(name){};
        virtual void generate_gradients(Graph* graph, NodeId current, std::unordered_map<NodeId, NodeId>& messages) = 0;
        virtual std::vector<std::weak_ptr<Node>> get_parents() = 0;
        virtual std::vector<std::weak_ptr<Node>> get_arguments() = 0;
        std::vector<std::weak_ptr<Node>> get_ancestors(){
            auto parents = this->get_parents();
            auto arguments = this->get_arguments();
            for(int i=0; i<arguments.size();i++){
                parents.push_back(arguments[i]);
            }
            return parents;
        }
    };

    class Node{
    public:
        std::string name;
        size_t id;
        TYPE type;
        af::array value;
        std::shared_ptr<Operator> op;
        std::vector<std::weak_ptr<Node>> children;
        int grad_level;
        Node() {}

        void dagre_print(std::string* nodes, std::string* edges, std::vector<NodeId> targets){
            std::string state_name;
            std::string state_color;
            if(this->type == INPUT) {
                state_name = "INPUT[" + std::to_string(this->id) + "]";
            } else if(this->type == CONST){
                state_name = "CONST[" + std::to_string(this->id) + "]";
            } else {
                state_name = this->op->name + "[" + std::to_string(this->id) + "]";
            }
            if(std::find(targets.begin(), targets.end(), this->id) != targets.end()) {
                state_color = "#f00";
            } else if(this->type == INPUT) {
                state_color = "#0f0";
            } else if(this->type == CONST){
                state_color = "#ff0";
            } else {
                state_color ="#00f";
            }
            auto ancestors = this->op->get_ancestors();
            std::string anc_id_str = "[";
            std::vector<std::string> anc_names;
            for(int i=0;i<ancestors.size();i++){
                auto anc = ancestors[i].lock();
                anc_id_str += std::to_string(anc->id);
                if(anc->type == INPUT) {
                    anc_names.push_back("INPUT[" + std::to_string(anc->id) + "]");
                } else if(anc->type == CONST){
                    anc_names.push_back("CONST[" + std::to_string(anc->id) + "]");
                } else {
                    anc_names.push_back(anc->op->name + "[" + std::to_string(anc->id) + "]");
                }
                if(i < ancestors.size()-1){
                    anc_id_str += ", ";
                }
            }
            anc_id_str += "]";
            nodes->append("\t'" + state_name + "': {\n"
                    "\t\tdescription: \"Name: " + this->name + " <br> "
                                  "Parents: " + anc_id_str + " <br> "
                                  "Gradient Level:" + std::to_string(this->grad_level) +  "\",\n"
                                  "\t\tstyle: \"fill: " + state_color + "\"\n"
                                  "\t},\n");
            for(int i=0;i<anc_names.size();i++){
                edges->append("g.setEdge({v:'" + anc_names[i] +  "', w:'" + state_name + "', name:'" + std::to_string(i) +
                                      "'}, {label: \"" + std::to_string(i) + "\"});\n");
            }
            edges->append("g.setParent('" + state_name + "', 'grad_" + std::to_string(this->grad_level) + "');\n");
        }

    };

    class InputOperator : public Operator {
    public:
        InputOperator(): Operator("IN"){}

        std::vector<std::weak_ptr<Node>> get_parents(){
            return std::vector<std::weak_ptr<Node>> {};
        }

        std::vector<std::weak_ptr<Node>> get_arguments(){
            return std::vector<std::weak_ptr<Node>> {};
        }

        void generate_gradients(Graph* graph, NodeId current, std::unordered_map<NodeId, NodeId>& messages){

        }
    };

    class Graph{
    public:
        std::vector<std::shared_ptr<Node>> nodes;
        Graph(){}

        NodeId create_input_node(){
            auto result = std::make_shared<Node>();
            result->op = std::make_shared<InputOperator>();
            result->id = nodes.size();
            result->name = "Input Node";
            result->type = INPUT;
            result->grad_level = 0;
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId create_constant_node(af::array value){
            auto result = std::make_shared<Node>();
            result->op = std::make_shared<InputOperator>();
            result->id = nodes.size();
            result->name = "Constant";
            result->type = CONST;
            result->value = value;
            result->grad_level = 0;
            this->nodes.push_back(result);
            return result->id;
        }

        NodeId create_derived_node(std::shared_ptr<Operator> op){
            auto result = std::make_shared<Node>();
            result->op = op;
            result->id = nodes.size();
            result->name = "Derived";
            result->type = INPUT_DERIVED;
            result->grad_level = 0;
            this->nodes.push_back(result);
            return result->id;
        }

        std::vector<NodeId> gradient(NodeId objective, std::vector<NodeId> params){
            std::unordered_map<NodeId, NodeId> grad_messages;
            auto target = this->nodes[objective];
            auto unity_grad = this->create_constant_node(af::constant(1.0, af::dim4(10, 10, 1, 1), f32));
            this->nodes[unity_grad]->grad_level = target->grad_level + 1;
            grad_messages[target->id] = unity_grad;
            this->nodes[grad_messages[target->id]]->name = "Grad of " + std::to_string(objective);
            long n = this->nodes.size();
            int j=0;
            for(auto i=n-1;i>=0;i--){
                if(grad_messages.find(i) != grad_messages.end()){
                    this->nodes[i]->op->generate_gradients(this, i, grad_messages);
                }
            }
            std::vector<NodeId> grads;
            for(int i=0;i<params.size();i++){
                grads.push_back(grad_messages[params[i]]);
            }
            return grads;
        }
        NodeId add(NodeId arg1, NodeId arg2);
        NodeId neg(NodeId arg1);
        NodeId mul(NodeId arg1_id, NodeId arg2_id);

        void print_to_file(std::string file_name, std::vector<NodeId> targets){
            int max_grad_level = 0;
            for(int i=0; i<targets.size(); i++){
                if(this->nodes[targets[i]]->grad_level > max_grad_level){
                    max_grad_level = this->nodes[targets[i]]->grad_level;
                }
            }
            std::ofstream f;
            f.open (file_name);
            std::string edges;
            std::string nodes;
            for(int i=0; i<this->nodes.size(); i++){
                this->nodes[i]->dagre_print(&nodes, &edges, targets);
            }
            f << ""
                    "<!DOCTYPE html>\n"
                    "<!-- saved from url=(0065)http://cpettitt.github.io/project/dagre-d3/latest/demo/hover.html -->\n"
                    "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"><meta charset=\"utf-8\">\n"
                    "<title>Function graph</title>\n"
                    "\n"
                    "<link rel=\"stylesheet\" href=\"../html_files/demo.css\">\n"
                    "<script src=\"../html_files/d3.v3.min.js\" charset=\"utf-8\"></script>\n"
                    "<script src=\"../html_files/dagre-d3.js\"></script><style type=\"text/css\"></style>\n"
                    "\n"
                    "<!-- Pull in JQuery dependencies -->\n"
                    "<link rel=\"stylesheet\" href=\"../html_files/tipsy.css\">\n"
                    "<script src=\"../html_files/jquery-1.9.1.min.js\"></script>\n"
                    "<script src=\"../html_files/tipsy.js\"></script>\n"
                    "\n"
                    "</head><body style=\"margin:0;padding:0\" height=\"100%\"><h3>Function graph</h3>\n"
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
            f << nodes;
            f << "}\n";
            "// Set the gradient level boxes\n";
            // Print grad boxes
            for(int i=0;i<=max_grad_level; i++){
                auto name = "Calculation Level[" + std::to_string(i) + "]";
                f << "g.setNode('grad_" + std::to_string(i) + "', {label: '" + name + "', clusterLabelPos: 'top', style: 'fill: #d3d7e8'});\n";
            }

            f << "// Add states to the graph, set labels, and style\n"
                    "Object.keys(states).forEach(function(state) {\n"
                    "  var value = states[state];\n"
                    "  value.label = state;\n"
                    "  value.rx = value.ry = 5;\n"
                    "  g.setNode(state, value);\n"
                    "});\n";

            // Print the edge connections
            f << "// Set up the edges\n";
            f << edges;
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
        }
    };


}

#endif //AUTODIFF_AUTODIFF_H_