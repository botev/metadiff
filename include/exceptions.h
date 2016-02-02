//
// Created by alex on 01/02/16.
//

#ifndef METADIFF_EXCEPTIONS_H
#define METADIFF_EXCEPTIONS_H
namespace metadiff{

    class UnsupportedGradient : public std::exception {
        const char* what() const throw(){
            return "\nThe gradient operation supports only scalar objectives";
        }
    };

    class GraphError : public std::exception {
    protected:
        std::string get_ids_and_shape_msg() const{
            std::string id_msg;
            std::string shape_msg;
            for(auto i=0;i<input_ids.size();i++){
                id_msg += std::to_string(input_ids[i]);
                shape_msg += "[";
                for(auto j=0; j<4; j++){
                    shape_msg += input_shapes[i][j].to_string();
                    if(j < 3){
                        shape_msg += ", ";
                    }
                }
                shape_msg += "]";
                if(i < input_ids.size() - 1){
                    id_msg += ", ";
                    shape_msg += ", ";
                }
            }
            return "Input ids: " + id_msg + "\nInput shapes: " + shape_msg + "\n";
        }
    public:
        std::string op_name;
        std::vector<size_t> input_ids;
        std::vector<Shape> input_shapes;
        std::string msg;
        GraphError(std::string op_name,
                   std::vector<size_t> input_ids,
                   std::vector<Shape> input_shapes):
                op_name(op_name),
                input_ids(input_ids),
                input_shapes(input_shapes) {};
        GraphError(std::string name, NodeVec inputs):
                op_name(op_name) {
            for(int i=0;i < inputs.size(); i++){
                input_ids.push_back(inputs[i].unwrap()->id);
                input_shapes.push_back(inputs[i].unwrap()->shape);
            }
        };

        const char* what() const throw(){
            return msg.c_str();
        }
    };

    class ImplicitBroadcast: public GraphError {
    private:
        std::string generate_message() const{
            std::string id_msg;
            std::string shape_msg;
            for(size_t i=0;i<input_ids.size();i++){
                id_msg += std::to_string(input_ids[i]);
                shape_msg += "[";
                for(size_t j=0; j<4; j++){
                    shape_msg += input_shapes[i][j].to_string();
                    if(j < 3){
                        shape_msg += ", ";
                    }
                }
                shape_msg += "]";
                if(i < input_ids.size() - 1){
                    id_msg += ", ";
                    shape_msg += ", ";
                }
            }
            return "\nImplicit broadcast in operator '" + op_name + "'\n" +
                   get_ids_and_shape_msg();
        }

    public:
        ImplicitBroadcast(std::string name,
                          std::vector<size_t> input_ids,
                          std::vector<Shape> input_shapes):
                GraphError(name, input_ids, input_shapes) {
            msg = generate_message();
        }

        ImplicitBroadcast(std::string name,
                          NodeVec inputs) :
                GraphError(name, inputs) {
            msg = generate_message();
        }
    };

    class OperatorError: public GraphError {
    private:
        std::string generate_message() const{
            return "\nOperator error in '" + op_name + "'\n" +
                   "Error message: " + err_string + "\n" +
                   get_ids_and_shape_msg();
        }
    public:
        std::string err_string;
        OperatorError(std::string op_name,
                      std::vector<size_t> input_ids,
                      std::vector<Shape> input_shapes,
                      std::string err_string):
                GraphError(op_name, input_ids, input_shapes),
                err_string(err_string) {
            msg = generate_message();
        };

        OperatorError(std::string op_name,
                      NodeVec inputs,
                      std::string err_string):
                GraphError(op_name, inputs),
                err_string(err_string) {
            msg = generate_message();
        };
    };


    class IncompatibleShapes: public GraphError {
    private:
        std::string generate_message() const{
            return "\nIncompatible dimensions in operator '" + op_name + "'\n" +
                   get_ids_and_shape_msg();
        }
    public:
        IncompatibleShapes(std::string name,
                           std::vector<size_t> input_ids,
                           std::vector<Shape> input_shapes):
                GraphError(name, input_ids, input_shapes) {
            msg = generate_message();
        }

        IncompatibleShapes(std::string name,
                           NodeVec inputs) :
                GraphError(name, inputs) {
            msg = generate_message();
        }
    };

    class InvalidArguments: public GraphError {
    private:
        std::string generate_message() const{
            return "\nInvalid arguments in operator '" + op_name + "'\n" +
                   "Arguments error: " + err_string + "\n";
            get_ids_and_shape_msg();
        }
    public:
        std::string err_string;
        InvalidArguments(std::string name,
                         std::vector<size_t> input_ids,
                         std::vector<Shape> input_shapes,
                         std::string err_string):
                GraphError(name, input_ids, input_shapes),
                err_string(err_string) {

        };

        InvalidArguments(std::string name,
                         NodeVec inputs,
                         std::string err_string):
                GraphError(name, inputs),
                err_string(err_string) {

        };
    };

    class WrongGradient: public GraphError {
    private:
        std::string generate_message() const{
            return "\nA gradient message to node " + std::to_string(input_ids[0]) +
                   " was sent, but all its parents are constant\n" +
                   "Node operator: '" + op_name + "'\n" +
                   "Message id: " + std::to_string(input_ids[1]) + "\n";
        }
    public:
        WrongGradient(std::string op_name,
                      std::vector<size_t> input_ids,
                      std::vector<Shape> input_shapes):
                GraphError(op_name, input_ids, input_shapes) {
            msg = generate_message();
        };

        WrongGradient(std::string op_name,
                      NodeVec inputs):
                GraphError(op_name, inputs) {
            msg = generate_message();
        };
    };
}

#endif //METADIFF_EXCEPTIONS_H
