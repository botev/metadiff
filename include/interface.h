//
// Created by alex on 20/12/15.
//

#ifndef METADIFF_INTERFACE_H
#define METADIFF_INTERFACE_H

#include <exception>

class InvalidInputShape : public std::exception {
public:
    size_t id;
    size_t expected[4];
    size_t given[4];
    std::string msg;

    InvalidInputShape(size_t id,
                      size_t  expected[4],
                      size_t  given[4]) :
            id(id){
        for(int i=0;i<4;i++){
            this->expected[i] = expected[i];
            this->given[i] = given[i];
        }
        msg = "The input node with id " + std::to_string(id) + " provided has incorrect shape.\n" +
              "Expected:" + std::to_string(expected[0]) + ", " + std::to_string(expected[1]) + ", "
              + std::to_string(expected[2]) + ", " + std::to_string(expected[3]) + ", " + "\n" +
              "Given:   " + std::to_string(given[0]) + ", " + std::to_string(given[1]) + ", "
              + std::to_string(given[2]) + ", " + std::to_string(given[3]) + ", " + "\n";
    };

    const char *what() const throw() {
        return msg.c_str();
    }
};

class SharedVariable{
public:
    size_t id;
    af::array value;
    SharedVariable():
            id(0),
            value(af::array())
    {};
    SharedVariable(size_t id, af::array value):
            id(id),
            value(value)
    {};
};
typedef std::shared_ptr<SharedVariable> SharedPtr;

#endif //METADIFF_INTERFACE_H
