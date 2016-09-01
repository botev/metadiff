//
// Created by alex on 08/05/16.
//

#ifndef METADIFF_DEFS_H
#define METADIFF_DEFS_H

namespace metadiff{
    namespace  core{
        /** The root NodeGroup name */
        static std::string const GROUP_ROOT  = "_root";

        /** The NodeGroup name separator */
        static char const GROUP_DELIMITER = '/';

        /**
         * When calling an Operator working along one axis
         * this flag for the axis indicates to auto infer it.
         * (Can we make this int8?)
         */
        static short const AUTO_INFER_AXIS = 1000;

        enum nodeType {
            /** The node represents a constant */
                    CONSTANT = 0,
            /** The node is derived from a constant, trough one or more Operator */
                    CONSTANT_DERIVED = 1,
            /** The node is an input. It can be either function input or a shared variable */
                    INPUT = 2,
            /** The node is derived from an input, trough one or more Operator */
                    INPUT_DERIVED = 3,
            /** The node is derived from an input, trough one or more Operator, but is Constant (no gradients) */
                    INPUT_DERIVED_CONSTANT = 4
        };

        /**
         * Data type of a Node
         *
         * Note that currently not all provided types are supported
         */
        enum dType{
            /** 8 bit boolean */
                    b8 = 0,
            /** 8 bit unsigned integer */
                    u8 = 1,
            /** 16 bit unsigned integer */
                    u16 = 2,
            /** 32 bit unsigned integer */
                    u32 = 3,
            /** 64 bit unsigned integer */
                    u64 = 4,
            /** 8 bit signed integer */
                    i8 = 5,
            /** 16 bit signed integer */
                    i16 = 6,
            /** 32 bit signed integer */
                    i32 = 7,
            /** 64 bit signed integer */
                    i64 = 8,
            /** 8 bit floating point */
                    f8 = 9,
            /** 16 bit floating point */
                    f16 = 10,
            /** 32 bit floating point */
                    f32 = 11,
            /** 64 bit floating point */
                    f64 = 12
        };

        /** Currently we support only two device types */
        enum deviceType {
            /** Represents a host with one or more CPUs */
                    HOST = 0,
            /** Represents a single GPU */
                    GPU = 1
        };

        /** An error policy defines how should we behave when an error occurs */
        enum errorPolicy {
            /** Does nothing */
                    QUIET = 0,
            /** Prints a warning */
                    WARN = 1,
            /** Throws an error */
                    RAISE = 2
        };

        /**
         * A single computational device to facilitate multy node computations
         * TODO not yet well designed, high probability it will change in the future
         */
        class Device {
        public:
            /** Type of the device - host or gpu*/
            deviceType type;
            /** A unique identifier of the device */
            size_t id;

            Device() :
                    type(HOST),
                    id(0) { };

            Device(deviceType type, size_t id) :
                    type(type),
                    id(id) { };

            Device(Device &device) :
                    type(device.type),
                    id(device.id) { };
        };

        /**
         * A NodeGroup is an abstraction of grouping together nodes (and groups).
         * How they are grouped is fully determinate by the user.
         * The hierarchy of groups is necessarily a DAG as well starting with a single root group.
         * The main goal of the groups is to provide a better way of visualizing the computation,
         * as well as a block for naming parameters accordingly.
         */
        class NodeGroup {
        public:
            /** The name of this group */
            std::string const name;
            /** This is the full name of the group, which depends on the parent */
            std::string full_name;
            /** The parent NodeGroup */
            std::weak_ptr<NodeGroup> const parent;
            /** The children groups */
            std::vector<std::weak_ptr<NodeGroup>> children;

            NodeGroup() :
                    name(GROUP_ROOT),
                    full_name(GROUP_ROOT) { };

            NodeGroup(std::string name,
                      std::weak_ptr<NodeGroup> parent) :
                    name(name),
                    parent(parent) {
                if (parent.lock()->full_name == GROUP_ROOT) {
                    full_name = name;
                } else {
                    full_name = parent.lock()->full_name;
                    full_name += GROUP_DELIMITER;
                    full_name += name;
                }
            };
        };

        // A few forward declarations and typdefs, unfortunately needed

        /** Axes are used for certain operators */
        typedef std::vector<short> Axes;
        /** A symbolic integer is just a SymbolicPolynomial */
        typedef symbolic::SymbolicPolynomial<unsigned short, unsigned short> SymInt;
        /**
        * The shape of any variable.
        * Currently we support 4 dimensional tensors.
        * Each dimension is a SymInt
        */
        typedef std::array<SymInt, 4> Shape;
        static const Shape scalar_shape = Shape{SymInt::one, SymInt::one, SymInt::one, SymInt::one};

        /** A group is a weak_ptr to internal Group */
        typedef std::weak_ptr<core::NodeGroup> Group;
        class GraphInternal;
        class NodeInternal;
        class Node;
        /** Vector of Nodes */
        typedef std::vector<Node> NodeVec;
        /** An update is a pair of shared variable and a node */
        typedef std::pair<Node, Node> Update;
        /** A collection of updates */
        typedef std::vector<std::pair<Node, Node>> Updates;
        /** Just a pointer to GraphInternal */
        typedef GraphInternal* GraphInPtr;
        /** A shared_ptr to GraphInternal, this is the outside API */
        typedef std::shared_ptr<core::GraphInternal> Graph;
        /** The host running the process has always an id of 0 */
        static const Device MASTER (HOST, 0);

        std::string to_string(nodeType node_type) {
            switch (node_type) {
                case CONSTANT:
                    return " Constant ";
                case CONSTANT_DERIVED:
                    return " ConstDer ";
                case INPUT :
                    return "   Input  ";
                case INPUT_DERIVED:
                    return "InDerived ";
                case INPUT_DERIVED_CONSTANT:
                    return "InDerConst";
                default:
                    return "UNREACHABLE";
            }
        }

        std::ostream &operator<<(std::ostream &f, nodeType node_type) {
            f << to_string(node_type);
            return f;
        }

        std::string to_string(dType dType) {
            if (dType == f64) {
                return "f64";
            } else if (dType == f32) {
                return "f32";
            } else if (dType == f16) {
                return "f16";
            } else if (dType == f8) {
                return "f8 ";
            } else if (dType == i64) {
                return "i64";
            } else if (dType == i32) {
                return "i32";
            } else if (dType == i16) {
                return "i16";
            } else if (dType == i8) {
                return "i8 ";
            } else if (dType == u64) {
                return "u64";
            } else if (dType == u32) {
                return "u32";
            } else if (dType == u16) {
                return "u16";
            } else if (dType == u8) {
                return "u8 ";
            } else if (dType == b8) {
                return "b8 ";
            } else {
                return "UNREACHABLE";
            }
        }

        std::ostream &operator<<(std::ostream &f, dType dType) {
            f << to_string(dType);
            return f;
        }

        std::string to_string(deviceType type) {
            switch (type) {
                case HOST:
                    return "HOST";
                case GPU:
                    return "GPU ";
                default:
                    return "UNREACHABLE";
            }
        }

        std::ostream &operator<<(std::ostream &f, deviceType deviceType) {
            f << to_string(deviceType);
            return f;
        }

        std::string to_string(errorPolicy policy) {
            switch (policy) {
                case RAISE:
                    return "Raise";
                case WARN:
                    return "Warn ";
                case QUIET:
                    return "Quiet";
                default:
                    return "UNREACHABLE";
            }
        }

        std::ostream &operator<<(std::ostream &f, errorPolicy policy) {
            f << to_string(policy);
            return f;
        }

        std::string to_string(Device const &device) {
            return to_string(device.type) + "[" + std::to_string(device.id) + "]";
        }

        std::ostream &operator<<(std::ostream &f, Device const &device) {
            f << to_string(device);
            return f;
        }

        std::string to_string(Shape const & shape){
            return "Shape(" + shape[0].to_string() + ", "
                   + shape[1].to_string() + ", "
                   + shape[2].to_string() + ", "
                   + shape[3].to_string() + ")";
        }

        std::ostream &operator<<(std::ostream &f, Shape const & shape){
            f << "(" << shape[0] << "," << shape[1] << "," << shape[2] << "," << shape[3] << ")";
            return f;
        }
    }
}

//namespace metadiff{
//    namespace core{
//        /** The maximum number of symbolic integers allowed */
////        static size_t const N = 1000;
//
//        /** The root NodeGroup name */
//        static std::string const GROUP_ROOT  = "_root";
//
//        /** The NodeGroup name separator */
//        static char const GROUP_DELIMITER = '/';
//
//        /**
//         * When calling an Operator working along one axis
//         * this flag for the axis indicates to auto infer it.
//         * (Can we make this int8?)
//         */
//        static short const AUTO_INFER_AXIS = 1000;
//
//        /** Each Node on the Graph is exactly one of these types */
//        enum nodeType {
//            /**
//    //         * The node is just a SymInt, which interacts with other nodes in an operator
//    //         */
////                SYMBOLIC_INTEGER,
//            /** The node represents a constant */
//                    CONSTANT = 0,
//            /** The node is derived from a constant, trough one or more Operator */
//                    CONSTANT_DERIVED = 1,
//            /**
//             * The node is an input.
//             * This can be either function input or a shared variable
//             */
//                    INPUT = 2,
////        /**
////         * The node is a shared variable
////         */
////                SHARED_INPUT,
//            /** The node is derived from an input, trough one or more Operator */
//                    INPUT_DERIVED = 3
//        };
//
//        /**
//         * Data type of a Node
//         *
//         * Note that currently not all provided types are supported
//         */
//        enum dType{
//            /** 8 bit boolean */
//                    b8 = 0,
//            /** 8 bit unsigned integer */
//                    u8 = 1,
//            /** 16 bit unsigned integer */
//                    u16 = 2,
//            /** 32 bit unsigned integer */
//                    u32 = 3,
//            /** 64 bit unsigned integer */
//                    u64 = 4,
//            /** 8 bit signed integer */
//                    i8 = 5,
//            /** 16 bit signed integer */
//                    i16 = 6,
//            /** 32 bit signed integer */
//                    i32 = 7,
//            /** 64 bit signed integer */
//                    i64 = 8,
//            /** 8 bit floating point */
//                    f8 = 9,
//            /** 16 bit floating point */
//                    f16 = 10,
//            /** 32 bit floating point */
//                    f32 = 11,
//            /** 64 bit floating point */
//                    f64 = 12
//        };
//
//        /** Currently we support only two device types */
//        enum deviceType {
//            /** Represents a host with one or more CPUs */
//                    HOST = 0,
//            /** Represents a single GPU */
//                    GPU = 1
//        };
//
//        /**
//         * A single computational device to facilitate multy node computations
//         * TODO not yet well designed, high probability it will change in the future
//         */
//        class Device {
//        public:
//            /** Type of the device - host or gpu*/
//            deviceType type;
//            /** A unique identifier of the device */
//            size_t id;
//
//            Device() :
//                    type(HOST),
//                    id(0) { };
//
//            Device(deviceType type, size_t id) :
//                    type(type),
//                    id(id) { };
//
//            Device(Device &device) :
//                    type(device.type),
//                    id(device.id) { };
//        };
//
//        /** An error policy defines how should we behave when an error occurs */
//        enum errorPolicy {
//            /** Does nothing */
//                    QUIET = 0,
//            /** Prints a warning */
//                    WARN = 1,
//            /** Throws an error */
//                    RAISE = 2
//        };
//    }
//}

//        enum numeric_type {
//            /**
//             * Represents floating point values
//             */
//                    FLOAT,
//            /**
//             * Represents integer values
//             */
//                    SIGNED_INTEGER,
//            /**
//             * Represents integer values
//             */
//                    UNSIGNED_INTEGER,
//            /**
//             * Represents boolean values
//             */
//                    BOOLEAN
//        };
//
//        enum bit_size {
//            /**
//             * 8 bit numeric
//             */
//                    bit8,
//            /**
//             * 16 bit numeric
//             */
//                    bit16,
//            /**
//             * 32 bit numeric
//             */
//                    bit32,
//            /**
//             * 64 bit numeric
//             */
//                    bit64
//        };


//        class ValueType {
//        public:
//            numeric_type num;
//            bit_size size;
//
//            ValueType(numeric_type num, bit_size size) :
//                    num(num), size(size) { };
//
//            ValueType() : ValueType(FLOAT, bit32) { };
//        };
//
//        static const ValueType f64 = ValueType(FLOAT, bit64);
//        static const ValueType f32 = ValueType(FLOAT, bit32);
//        static const ValueType f16 = ValueType(FLOAT, bit16);
//        static const ValueType f8 = ValueType(FLOAT, bit8);
//        static const ValueType i64 = ValueType(SIGNED_INTEGER, bit64);
//        static const ValueType i32 = ValueType(SIGNED_INTEGER, bit32);
//        static const ValueType i16 = ValueType(SIGNED_INTEGER, bit16);
//        static const ValueType i8 = ValueType(SIGNED_INTEGER, bit8);
//        static const ValueType u64 = ValueType(UNSIGNED_INTEGER, bit64);
//        static const ValueType u32 = ValueType(UNSIGNED_INTEGER, bit32);
//        static const ValueType u16 = ValueType(UNSIGNED_INTEGER, bit16);
//        static const ValueType u8 = ValueType(UNSIGNED_INTEGER, bit8);
//        static const ValueType b8 = ValueType(BOOLEAN, bit8);
//        bool operator==(ValueType const &type1, ValueType const &type2);

//    enum ad_float_type {
//        /**
//         * 16 bit floating point number
//         */
//                f16,
//        /**
//         * 32 bit floating point number
//         */
//                f32,
//        /**
//         * 64 bit floating point number
//         */
//                f64,
//    };
//    enum ad_integer_type {
//        /**
//         * 8 bit integer
//         */
//                i8,
//        /**
//        * 16 bit integer
//        */
//                i16,
//        /**
//        * 32 bit integer
//        */
//                i32,
//        /**
//        * 64 bit integer
//        */
//                i64,
//    };

#endif //METADIFF_DEFS_H
