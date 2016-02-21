var svgNS = "http://www.w3.org/2000/svg";
var PrimalNode = function(parent_group, name, node_info) {
    this.parent_group = parent_group;
    this.name = name;
    this.node_info = node_info;
    this.child_groups = [];
    this.child_nodes = [];
    this.collapsed = true;
    this.hidden = true;
    this.display_node = undefined;
    if (parent_group) {
        if (name[0] == '_') {
            parent_group.child_groups.push(this);
        } else {
            parent_group.child_nodes.push(this);
        }
    }

    this.getAllParentGroups = function (){
        if(this.parent_group) {
            var groups = [this.parent_group];
            while (groups[groups.length - 1].parent_group) {
                groups.push(groups[groups.length - 1].parent_group);
            }
            return groups;
        } else {
            return [];
        }
    };

    this.parentOf = function (node) {
        var all_parents = node.getAllParentGroups();
        for(var i=0; i<all_parents.length; i++){
            if(this.name == all_parents[i].name){
                return true;
            }
        }
        return false;
    };
};

var GraphMorpher = function(svg_tag, min_width, min_height) {
    this.delimiter = '/';
    this.svg_tag = svg_tag;
    this.min_width = min_width;
    this.min_height = min_height;
    this.render = new dagreD3.render();
    this.prime_graph = new dagreD3.graphlib.Graph({compound: true, multigraph: true})
        .setGraph({})
        .setDefaultEdgeLabel(function () {
            return {};
        });
    this.display_graph = new dagreD3.graphlib.Graph({compound: true, multigraph: true})
        .setGraph({})
        .setDefaultEdgeLabel(function () {
            return {};
        });

    this.prime_graph.setNode("_root", new PrimalNode(undefined, "_root", undefined));
    this.prime_graph.node("_root").hidden = false;

    this.addGroup = function (full_name) {
        var names = full_name.split(this.delimiter);
        var node = this.prime_graph.node(names[0]);
        var name_acc = names[0];
//            var w = this.min_width;
//            var h = this.min_height;
        for (var i = 1; i < names.length; i++) {
            name_acc += this.delimiter + names[i];
            for (var j = 0;  j < node.child_groups.length; j++){
                if(name_acc == node.child_groups[j].name){
                    break;
                }
            }
            if (j < node.child_groups.length) {
                node = node.child_groups[j];
            } else {
                this.prime_graph.setNode(name_acc, new PrimalNode(node, name_acc, {
                    description: full_name,
                    label: names[names.length-1] ,
                    clusterLabelPos: "top",
                    style: "fill: #A1A1A1; stroke:#454545;",
                    labelStyle: "font-weight: bold; font-size:1em"}));
                node = this.prime_graph.node(name_acc);
            }
        }
        return node;
    };

    this.addNode = function (group_name, name, node_info) {
        node_info.margin = 5;
        var parent_group = this.prime_graph.node(group_name);
        this.prime_graph.setNode(name, new PrimalNode(parent_group, name, node_info));
        return this.prime_graph.node(name);
    };

    this.addEdge = function (name_from, name_to, label) {
        var from_groups = this.prime_graph.node(name_from).getAllParentGroups();
        var to_groups = this.prime_graph.node(name_to).getAllParentGroups();
        // Set the original edge
        this.prime_graph.setEdge({v: name_from, w: name_to, name: label}, {label: label});
        for (var j = 0; j < to_groups.length; j++) {
            var to_group = to_groups[j];
            for (var i = 0; i < from_groups.length; i++) {
                var from_group = from_groups[i];

                if (!(from_group.parentOf(to_group)) && !(to_group.parentOf(from_group))
                    && !(from_group.name == to_group.name)) {
//                        console.log("Edge between", from_group.name, to_group.name);
//                    console.log(from_group.parentOf(to_group));
//                    console.log(to_group.parentOf(from_group));
                    this.prime_graph.setEdge(from_group.name, to_group.name);
                } else {
                    //                    console.log("No");
                    //                    console.log(from_group.parentOf(to_group));
                    //                    console.log(to_group.parentOf(from_group));
                    //                    console.log(from_group == to_group);
                }
                if(!(from_group.parentOf(this.prime_graph.node(name_to)))){
                    this.prime_graph.setEdge({v: from_group.name, w: name_to, name: ""}, {
                        lineInterpolate: "basis",
                        label: label
                    });
                }
            }
            if (!(to_group.parentOf(this.prime_graph.node(name_from)))) {
                this.prime_graph.setEdge({v: name_from, w: to_group.name, name: ""}, {
                    lineInterpolate: "basis",
                    label: label
                });
            }
        }
    };

    this.expandNode = function (node_name){
        var node = this.prime_graph.node(node_name);
        if(!node.hidden && node.collapsed){
            node.collapsed = false;
            node.hidden = false;
            var edges = [];
            if(node.display_node){
                this.display_graph.removeNode(node.name);
                this.display_graph.setNode(node.name, node.node_info);
                if(node.parent_group.display_node) {
                    this.display_graph.setParent(node.name, node.parent_group.name);
                }
                node.display_node = this.display_graph.node(node.name);
                node.node_info.style = "fill: #E2E2E2; stroke:#454545; font-weight: bold";
            }
//                console.log(node.child_groups);
            // Make special
//                if (node.display_node) {
//                    this.display_graph.setNode(node.name + "button", {
//                        description: "X",
//                        label: "X",
//                        shape: "ellipse"
//                    });
//                    this.display_graph.setParent(node.name + "button", node.name);
//                }
            // Make group nodes
            for (var i=0; i<node.child_groups.length; i++){
                var child = node.child_groups[i];
//                    console.log(child.node_info);
                this.display_graph.setNode(child.name, child.node_info);
                var display = this.display_graph.node(child.name);
                if(node.display_node) {
//                        console.log("Parent", child.name, node.name);
                    this.display_graph.setParent(child.name, node.name);
                }
                child.display_node = display;
                child.hidden = false;
                edges = edges.concat(this.prime_graph.nodeEdges(child.name));
            }
            // Make direct nodes
            for (var i=0; i<node.child_nodes.length; i++){
                var child = node.child_nodes[i];
                this.display_graph.setNode(child.name, child.node_info);
                var display = this.display_graph.node(child.name);
                if(node.display_node) {
//                        console.log("Parent", child.name, node.name);
                    this.display_graph.setParent(child.name, node.name);
                }
                child.display_node = display;
                child.hidden = false;
                edges = edges.concat(this.prime_graph.nodeEdges(child.name));
            }
            // Make all edges for which the nodes have been drawn
            for (var i=0; i<edges.length; i++){
                var from_node = this.prime_graph.node(edges[i].v);
                var to_node = this.prime_graph.node(edges[i].w);
                if(!from_node.hidden && from_node.collapsed &&
                    !to_node.hidden && to_node.collapsed){
//                        console.log(edges[i]);
                    this.display_graph.setEdge({v: edges[i].v, w: edges[i].w, name: edges[i].name}, {
                        lineInterpolate: "basis",
                        label: edges[i].name
                    });
                }
            }
        } else {
            console.log("Trying to expand a hidden or already expanded node.");
        }
    };

    this.collapseNode = function(node_name){
        var node = this.prime_graph.node(node_name);
        if(!node.hidden && !node.collapsed){
            node.collapsed = true;
            var inEdges = [];
            var outEdges = [];
            node.node_info.style = "fill: #A1A1A1; stroke:#454545; font-weight: bold";
            //console.log("I", node.node_info.style);
            //this.display_graph.removeNode(node.display_node);
            //this.display_graph.setNode(node.name, node.display_node);
            //console.log(node.node_info.style);
            // Remove all group nodes
            for (var i=0; i<node.child_groups.length; i++){
                var child = node.child_groups[i];
//                    console.log("Hiding", child.name, child.collapsed, child.node_info);
                if(! child.collapsed){
                    this.collapseNode(child.name);
                }
//                    console.log(child.node_info);
                var child_in = this.display_graph.inEdges(child.name);
                for(var j=0; j<child_in.length; j++){
                    inEdges.push(child_in[j].v);
                }
                var child_out = this.display_graph.outEdges(child.name);
                for(var j=0; j<child_out.length; j++){
                    outEdges.push(child_out[j].w);
                }
                this.display_graph.removeNode(child.name);
                child.display_node = undefined;
                child.hidden = true;
            }
            // Remove all child nodes
            for (var i=0; i<node.child_nodes.length; i++){
                var child = node.child_nodes[i];
//                    console.log("Hiding", child.name, child.collapsed, child.node_info);
                var child_in = this.display_graph.inEdges(child.name);
                for(var j=0; j<child_in.length; j++){
                    inEdges.push(child_in[j].v);
                }
                var child_out = this.display_graph.outEdges(child.name);
                for(var j=0; j<child_out.length; j++){
                    outEdges.push(child_out[j].w);
                }
                this.display_graph.removeNode(child.name);
                child.display_node = undefined;
                child.hidden = true;
            }
            // Create all of the edges required
            for(var i=0;i < inEdges.length ; i++){
                if(!this.prime_graph.node(inEdges[i]).hidden) {
//                        console.log("Edge in", node.name, inEdges[i]);
                    this.display_graph.setEdge({v: inEdges[i], w: node.name, name: "da"}, {
                        lineInterpolate: "basis",
                        label: ""});
                }
            }
            for(var i=0;i < outEdges.length ; i++){
                if(!this.prime_graph.node(outEdges[i]).hidden) {
//                        console.log("Edge out", node.name, outEdges[i]);
                    this.display_graph.setEdge({w: outEdges[i], v: node.name, name: "da"}, {
                        lineInterpolate: "basis",
                        label: ""});
                }
            }
        } else {
            console.log("Trying to collapse a hidden or already collapsed node.");
        }
    };

    this.expandOrCollapse = function (node_name){
        if(node_name[0] == "_"){
            var node = this.prime_graph.node(node_name);
            if(node.collapsed){
                this.expandNode(node_name);
            } else {
                this.collapseNode(node_name);
            }
            this.updateSVG();
        } else {
            console.log("Trying to expand or collapse a non group node.")
        }
    };

    this.updateSVG = function (){
        var inner = d3.select("svg")[0];
        this.display_graph.margix = 100;
        var morpher = this;
        var g = this.display_graph;
//            d3.selectAll("rect").style("width", 20);


        if (!g.graph().hasOwnProperty("marginx") &&
            !g.graph().hasOwnProperty("marginy")) {
            g.graph().marginx = 20;
            g.graph().marginy = 20;
        }
        g.graph().transition = function(selection) {
            return selection.transition().duration(500);
        };

        d3.select("svg g").call(this.render, g);


        d3.select("svg g").selectAll("g.cluster, g.node")
            .filter(function (v){
                return v[0] == "_";
            })
            .forEach(function(node){
                var add_node = function(node){
                    return function() {
//                                if (node.children.length == 2) {
                        var x = node.children[0].height.animVal.value;
                        var y = node.children[0].width.animVal.value;
                        var r = 8;
                        var myCircle = document.createElementNS(svgNS, "circle");
                        myCircle.setAttributeNS(null, "id", "mycircle");
                        myCircle.setAttributeNS(null, "cx", y / 2 - r);
                        myCircle.setAttributeNS(null, "cy", -x / 2 + r);
                        myCircle.setAttributeNS(null, "r", r);
                        myCircle.setAttributeNS(null, "fill", "#ffff00");
                        myCircle.setAttributeNS(null, "stroke", "none");
                        node.appendChild(myCircle);
                        myCircle.onclick = function (){
                            morpher.expandOrCollapse(node.__data__);
                        };
//                                }
                    };
                };
                for(var j=0;j<node.length; j++) {
                    setTimeout(add_node(node[j]),550);
                }
            });
            //.select("g.label")
            //.attr("transform", function(node, i){
            //    console.log(node, i);
            //    return "translate(-20, 0)";
            //});
//                    .append("children", function(v, i){
//
//                    });
//                    .apply(function (v){
//                        console.log("once", v);
//                    })
//                    .on('mouseenter', function(v) {
//                        var x = this.children[0].height.animVal.value;
//                        var y = this.children[0].width.animVal.value;
//                        var r = 10;
//                        var myCircle = document.createElementNS(svgNS,"circle");
//                        myCircle.setAttributeNS(null,"id","mycircle");
//                        myCircle.setAttributeNS(null,"cx",y/2 - r);
//                        myCircle.setAttributeNS(null,"cy",-x/2 + r);
//                        myCircle.setAttributeNS(null,"r",r);
//                        myCircle.setAttributeNS(null,"fill","black");
//                        myCircle.setAttributeNS(null,"stroke","none");
//                        this.appendChild(myCircle);
////                        console.log(v);
////                        console.log(morpher.prime_graph.node(v));
//                    })
//                    .on('mouseout', function(v) {
////                        this.select("circle").remove();
////                        console.log(this.children);
//                        console.log("out");
////                        this.children[2].remove();
////                        console.log(morpher.prime_graph.node(v));
//                    })
//                    .on('mouseleave', function(v) {
////                        this.select("circle").remove();
////                        console.log(this.children);
//                        console.log("Leave");
////                        this.children[2].remove();
////                        console.log(morpher.prime_graph.node(v));
//                    });
        d3.select("svg g").selectAll("rect").attr("rx", 5);
        d3.select("svg g").selectAll("rect").attr("ry", 5);

        // Simple function to style the tooltip for the given node.
        var styleTooltip = function (name, description) {
            return "<p class='name'>" + name + "</p><p class='description'  style=\"text-align:left\">" + description + "</p>";
        };

        d3.select("svg g").selectAll("g.node")
            .filter(function (v) {
                return !morpher.prime_graph.node(v).hidden && v[0] != '_';
            })
            .attr("title", function (v) {
                //console.log(v);
                return styleTooltip(v, g.node(v).description)
            })
            .each(function (v) {
                $(this).tipsy({gravity: "w", opacity: 1, html: true});
            });

        var t = function () {
            d3.select("svg g").selectAll("g.cluster").select("g.label")
                .attr("transform", function () {
                    //console.log($(this).children()[0].width);
                    var w = $(this).parents()[0].children[0].width.animVal.value;
                    //console.log(w/2);
                    //console.log(w);
                    return "translate(-" + (w/2+$(this).children()[0].transform.animVal[0].matrix.e - 5) + ", 0)";
                });
        };
        setTimeout(t, 500);
            //.forEach(function (v){
            //    console.log(v);
            //    for(var i=0; i< v.length; i++) {
            //        v[i].children[1].transform = "translate(20, 20)";
            //        console.log(v[i]    , i);
            //    }
            //});

//            d3.selectAll("rect").attr("x", function (d){
//                var w = this.width.animVal.value;
//                var x = this.x.animVal.value;
//                if(w < morpher.min_width){
//                    var diff = (morpher.min_width - w) / 2;
//                    return x - diff;
//                } else{
//                    return x;
//                }
//            });
//            d3.selectAll("rect").attr("width", function (d){
//                var val = this.width.animVal.value;
//                if(val < morpher.min_width){
//                    var diff = (morpher.min_width - val) / 2;
//                    return morpher.min_width;
//                } else{
//                    return val;
//                }
//            });
//            console.log("render");
    };

    this.populateSVG = function (){
        morpher.expandNode("_root");

        var g = this.display_graph;
        var render = this.render;

        // Set up an SVG group so that we can translate the final graph.
        var svg = d3.select("svg"),
            inner = svg.append(svg_tag);

        var zoom = d3.behavior.zoom().on("zoom", function () {
            inner.attr("transform", "translate(" + d3.event.translate + ")" +
                "scale(" + d3.event.scale + ")");
        });
        svg.call(zoom);
        // Run the renderer. This is what draws the final graph.
        render(inner, g);
        var initialScale = 0.75;
        zoom
            .translate([(svg.attr("width") - g.graph().width * initialScale) / 2, 20])
            .scale(initialScale)
            .event(svg);
        //svg.attr('height', Math.max(g.graph().height * initialScale + 40, this.min_height));
        this.updateSVG();
    };

//        this.generate = function () {
//            var root = this.prime_graph.node("_root");
//            root.collapsed = false;
//            root.hidden = false;
//            var edges = [];
//            // Make group nodes
//            for (var i=0; i<root.child_groups.length; i++){
//                var node = root.child_groups[i];
//                this.display_graph.setNode(node.name, node.node_info);
//                node.display_node = this.display_graph[node.name];
//                node.hidden = false;
//                edges.concat(this.prime_graph.outEdges(node.name));
//            }
//            // Make direct nodes
//            for (var i=0; i<root.child_nodes.length; i++){
//                var node = root.child_nodes[i];
//                this.display_graph.setNode(node.name, node.node_info);
//                node.display_node = this.display_graph[node.name];
//                node.hidden = false;
//                edges = edges.concat(this.prime_graph.outEdges(node.name));
//            }
//            // Make all edges for which the nodes have been drawn
//            for (var i=0; i<edges.length; i++){
//                var to_node = this.prime_graph.node(edges[i].w);
//                if(! to_node.hidden){
//                    console.log(edges[i]);
//                    this.display_graph.setEdge({v: edges[i].v, w: edges[i].w, name: edges[i].name}, {label: edges[i].name});
//                }
//            }
//        }
};