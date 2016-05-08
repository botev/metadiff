//
// Created by alex on 08/05/16.
//

#ifndef METADIFF_SHARED_H
#define METADIFF_SHARED_H

namespace metadiff{
    namespace shared{
        /** A shared variable is a like a static variable, which is synchronized between devices */
        class SharedVariable {
        public:
            size_t const id;
            std::string const name;
            std::array<long long, 4> const shape;

            af::array value;
        public:
            SharedVariable(size_t id,
                           std::array<long long, 4> shape,
                           std::string name):
                    id(id),
                    shape(shape),
                    name(name) {};

            SharedVariable(size_t id,
                           af::array value,
                           std::string name):
                    id(id),
                    value(value),
                    shape(std::array<long long, 4> {value.dims(0), value.dims(1),
                                                    value.dims(2), value.dims(3)}),
                    name(name) {};
        };

        typedef std::shared_ptr<SharedVariable> SharedPtr;
        static std::vector<SharedPtr> shared_vars;

        static SharedPtr make_shared(af::array value, std::string name){
            SharedPtr ptr = std::make_shared<SharedVariable>(shared_vars.size(), value, name);
            shared_vars.push_back(ptr);
            return ptr;
        }
    }
}


#endif //METADIFF_SHARED_H
