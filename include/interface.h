//
// Created by alex on 20/12/15.
//

#ifndef METADIFF_INTERFACE_H
#define METADIFF_INTERFACE_H

namespace metadiff{
    namespace core {
        enum dType {
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
    }

    namespace shared{
        /** A shared variable is a like a static variable, which is synchronized between devices */
        class SharedVariable {
        public:
            size_t const id;
            std::string const name;
            std::array<long long, 4> const shape;

//            af::array value;
        public:
            SharedVariable(size_t id,
                           std::array<long long, 4> shape,
                           std::string name):
                    id(id),
                    shape(shape),
                    name(name) {};

            virtual core::dType get_dtype() const = 0;
        };

        typedef std::shared_ptr<SharedVariable> SharedPtr;
        static std::vector<SharedPtr> shared_vars;

        /** A shared variable is a like a static variable, which is synchronized between devices */
        class ArrayFireVariable: public SharedVariable {
        public:
            af::array value;
            ArrayFireVariable(size_t id,
                              af::array value,
                              std::string name):
                    SharedVariable(id, std::array<long long, 4> {value.dims(0), value.dims(1),
                                                                 value.dims(2), value.dims(3)},
                                   name),
                    value(value) {};

            /** Converts an af_dtype to dType
             * TODO: Make proper exception when given complex type */
            static core::dType convert_af_dtype(af_dtype dtype){
                switch (dtype){
                    case af_dtype::b8 : return core::b8;
                    case af_dtype::u8 : return core::u8;
                    case af_dtype::u16: return core::u16;
                    case af_dtype::u32: return core::u32;
                    case af_dtype::u64: return core::u64;
                    case af_dtype::s16: return core::i16;
                    case af_dtype::s32: return core::i32;
                    case af_dtype::s64: return core::i64;
                    case af_dtype::f32 : return core::f32;
                    case af_dtype::f64: return core::f64;
                    default: throw 20;
                }
            }

            core::dType get_dtype() const{
                return ArrayFireVariable::convert_af_dtype(value.type());
            }
        };

        typedef std::shared_ptr<ArrayFireVariable> AfShared;

        static SharedPtr make_shared(af::array value, std::string name){
            SharedPtr ptr = std::make_shared<ArrayFireVariable>(shared_vars.size(), value, name);
            shared_vars.push_back(ptr);
            return ptr;
        }
    }
}

using metadiff::shared::SharedVariable;
using metadiff::shared::SharedPtr;
using metadiff::shared::ArrayFireVariable;
using metadiff::shared::AfShared;

#endif //METADIFF_INTERFACE_H
