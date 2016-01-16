//
// Created by alex on 16/01/16.
//

#ifndef METADIFF_TEST_H
#define METADIFF_TEST_H
//namespace metadiff{
//    const size_t N = 100;
//
//    typedef symbolic::SymbolicPolynomial<N, unsigned short> SymInt;
//    typedef std::array<SymInt,4> Shape;
//
//    enum ad_node_type{SYMBOLIC_INTEGER, CONSTANT, INPUT, SHARED_INPUT, INPUT_DERIVED, CONSTANT_DERIVED, UPDATE};
//    enum ad_value_type{FLOAT, INTEGER, BOOLEAN};
//    enum ad_device_type {CPU, GPU};
//    enum ad_implicit_broadcast {RAISE, WARN, QUIET};
//    enum ad_float_type {f16, c16, f32, c32, f64, c64};
//    enum ad_integer_type {s8, u8, s16, u16, s32, u32, s64, u64};
//
//    class GraphInternal;
//    typedef GraphInternal* GraphInPtr;
//    typedef std::shared_ptr<GraphInternal> Graph;
//    class Operator;
//    class NodeInternal;
//    typedef std::vector<NodeInternal*> NodeVec;
//
//    class Device{
//    public:
//        ad_device_type type;
//        size_t id;
//        Device():
//                type(ad_device_type::CPU),
//                id(0)
//        {};
//
//        Device(const ad_device_type type, const size_t id):
//                type(type),
//                id(id)
//        {};
//    };
//
//    class ExecutionData{
//    public:
//        bool inlined;
//        size_t register_id;
//        size_t lifespan;
//        ExecutionData():
//                inlined(false),
//                register_id(0),
//                lifespan(0)
//        {}
//    };
//
//    class NodeInternal{
//    public:
//        GraphInPtr graph;
//        Device device;
//        size_t id;
//        std::string name;
//        ad_node_type type;
//        ad_value_type v_type;
//        Shape shape;
//        std::shared_ptr<Operator> op;
//        NodeVec children;
//        size_t grad_level;
//        af::array value;
//        SharedPtr shared;
//
//        ExecutionData execution;
//
//        NodeInternal(GraphInPtr graph, Device device):
//                graph(graph),
//                device(device)
//        {}
//
//        NodeInternal(GraphInPtr graph,
//                     Device device,
//                     size_t id,
//                     std::string name,
//                     ad_node_type type,
//                     ad_value_type v_type,
//                     Shape shape,
//                     std::shared_ptr<Operator> op,
//                     size_t grad_level):
//                graph(graph),
//                device(device),
//                id(id),
//                name(name),
//                type(type),
//                v_type(v_type),
//                op(op),
//                grad_level(grad_level),
//                shape(shape)
//        {}
//    };
//
//    class Node{
//        NodeInternal* ptr;
//        Node() {};
//    };
//}
#endif //METADIFF_TEST_H
