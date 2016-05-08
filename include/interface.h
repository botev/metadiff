//
// Created by alex on 20/12/15.
//

#ifndef METADIFF_INTERFACE_H
#define METADIFF_INTERFACE_H

#include <exception>

namespace metadiff{
    namespace core{

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

        /** A shared variable is a like a static variable, which is synchronized between devices */
        class SharedVariable {
        public:
            static std::vector<std::shared_ptr<SharedVariable>> shared_vars;
            size_t const id;
            dType const dtype;
            std::string const name;
            std::array<size_t, 4> const shape;

//            template <typename T>
//            T* get_device_pointer(Device device);
//
//            template <typename T>
//            T* get_pointer(){
//                return get_device_pointer<T>(MASTER);
//            }
        protected:
            SharedVariable(std::array<size_t, 4> shape, std::string name):
                    id(shared_vars.size()),
                    dtype(dtype),
                    shape(shape),
                    name(name) {};
        };

        typedef std::shared_ptr<SharedVariable> SharedPtr;
        std::vector<SharedPtr> SharedVariable::shared_vars;
    }

    namespace exceptions{
        class InvalidInputShape : public std::exception {
        private:
            std::string generate_message(){
                std::stringstream msg;
                msg << "The input to the function at index " << input_index << "(zero based), "
                << "corresponding to node with id " << id << " has expected shape"
                << "(" << expected_shape[0] << "," << expected_shape[1] << ","
                << expected_shape[2] << "," << expected_shape[3] << "), "
                << "but the provided input had shape"
                << "(" << provided_shape[0] << "," << provided_shape[1] << ","
                << provided_shape[2] << "," << provided_shape[3] << ").";
                return msg.str();
            }
        public:
            size_t input_index;
            size_t id;
            std::array<size_t, 4> expected_shape;
            std::array<size_t, 4> provided_shape;
            std::string msg;
            InvalidInputShape(): msg("") {};

            InvalidInputShape(size_t const input_index,
                              size_t const id,
                              std::array<size_t, 4> expected_shape,
                              std::array<size_t, 4> provided_shape):
                    input_index(input_index), id(id),
                    expected_shape(expected_shape),
                    provided_shape(provided_shape),
                    msg(generate_message()) {};

            const char *what() const throw() {
                return msg.c_str();
            }
        };
    }
}

using metadiff::core::SharedPtr;
using metadiff::exceptions::InvalidInputShape;

#endif //METADIFF_INTERFACE_H
