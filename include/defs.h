//
// Created by alex on 08/05/16.
//

#ifndef METADIFF_DEFS_H
#define METADIFF_DEFS_H

namespace metadiff{
    namespace core{
        /** The maximum number of symbolic integers allowed */
        static size_t const N = 1000;

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

        /** Each Node on the Graph is exactly one of these types */
        enum nodeType {
            /**
    //         * The node is just a SymInt, which interacts with other nodes in an operator
    //         */
//                SYMBOLIC_INTEGER,
            /** The node represents a constant */
                    CONSTANT = 0,
            /** The node is derived from a constant, trough one or more Operator */
                    CONSTANT_DERIVED = 1,
            /**
             * The node is an input.
             * This can be either function input or a shared variable
             */
                    INPUT = 2,
//        /**
//         * The node is a shared variable
//         */
//                SHARED_INPUT,
            /** The node is derived from an input, trough one or more Operator */
                    INPUT_DERIVED = 3
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

        /** An error policy defines how should we behave when an error occurs */
        enum errorPolicy {
            /** Does nothing */
                    QUIET = 0,
            /** Prints a warning */
                    WARN = 1,
            /** Throws an error */
                    RAISE = 2
        };

    }
}

#endif //METADIFF_DEFS_H
