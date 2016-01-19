//
// Created by alex on 18/01/16.
//
#define EIGEN_USE_MKL_ALL
#include "Eigen/Core"
#include <sys/stat.h>
#include "curl/curl.h"
#include "vector"
#include "iostream"
#include <fstream>
#include "memory"

const char kPathSeparator =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif

struct FtpFile {
    const char *filename;
    FILE *stream;
};

static size_t my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream);
static size_t my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream);
void download_mnist(std::string folder);
int ReverseInt (int i);
void ReadTrainMNIST(std::string folder, float* data, int* labels);

inline Eigen::MatrixXf softplus(Eigen::MatrixXf input, int threshold) {
    return (input.array().exp()+1).array().log();
}

std::string size(Eigen::Ref<Eigen::MatrixXf> in){
    return "(" + std::to_string(in.rows()) + ", " + std::to_string(in.cols()) + ")";
}

const int num_images = 60000;
const int rows = 28;
const int cols = 28;

std::vector<Eigen::MatrixXf> eval_func(std::vector<Eigen::MatrixXf>& inputs, std::vector<Eigen::MatrixXf>& shared_vars){

    // Calculate all of the computation nodes
    Eigen::MatrixXf node_20 = ((inputs[0] * shared_vars[0]) + shared_vars[1].transpose().replicate(1000,1)).unaryExpr(std::ptr_fun(tanhf));
    Eigen::MatrixXf node_24 = ((node_20 * shared_vars[2]) + shared_vars[3].transpose().replicate(1000,1)).unaryExpr(std::ptr_fun(tanhf));
    Eigen::MatrixXf node_28 = ((node_24 * shared_vars[4]) + shared_vars[5].transpose().replicate(1000,1)).unaryExpr(std::ptr_fun(tanhf));
    Eigen::MatrixXf node_32 = ((node_28 * shared_vars[6]) + shared_vars[7].transpose().replicate(1000,1)).unaryExpr(std::ptr_fun(tanhf));
    Eigen::MatrixXf node_36 = ((node_32 * shared_vars[8]) + shared_vars[9].transpose().replicate(1000,1)).unaryExpr(std::ptr_fun(tanhf));
    Eigen::MatrixXf node_40 = ((node_36 * shared_vars[10]) + shared_vars[11].transpose().replicate(1000,1)).unaryExpr(std::ptr_fun(tanhf));
    Eigen::MatrixXf node_44 = ((node_40 * shared_vars[12]) + shared_vars[13].transpose().replicate(1000,1)).unaryExpr(std::ptr_fun(tanhf));
    Eigen::MatrixXf node_47 = (node_44 * shared_vars[14]) + shared_vars[15].transpose().replicate(1000,1);
    Eigen::MatrixXf node_59 = 0.001000 * ((1.0 / (1.0 + (-node_47).array().exp()).array()).matrix() - inputs[0]);

    Eigen::MatrixXf node_69 = (node_59 * shared_vars[14].transpose()).array() * (1.000000 - node_44.array().square());
    Eigen::MatrixXf node_79 = (node_69 * shared_vars[12].transpose()).array() * (1.000000 - node_40.array().square());
    Eigen::MatrixXf node_89 = (node_79 * shared_vars[10].transpose()).array() * (1.000000 - node_36.array().square());
    Eigen::MatrixXf node_99 = (node_89 * shared_vars[8].transpose()).array() * (1.000000 - node_32.array().square());
    Eigen::MatrixXf node_109 = (node_99 * shared_vars[6].transpose()).array() * (1.000000 - node_28.array().square());
    Eigen::MatrixXf node_119 = (node_109 * shared_vars[4].transpose()).array() * (1.000000 - node_24.array().square());
    Eigen::MatrixXf node_129 = (node_119 * shared_vars[2].transpose()).array() * (1.000000 - node_20.array().square());

    // Update all shared variables
    shared_vars[0].noalias() -= 0.010000 * (inputs[0].transpose() * node_129);
    shared_vars[1].noalias() -= 0.010000 * node_129.colwise().sum().transpose();
    shared_vars[2] -= 0.010000 * (node_20.transpose() * node_119);
    shared_vars[3] -= 0.010000 * node_119.colwise().sum().transpose();
    shared_vars[4] -= 0.010000 * (node_24.transpose() * node_109);
    shared_vars[5] -= 0.010000 * node_109.colwise().sum().transpose();
    shared_vars[6] -= 0.010000 * (node_28.transpose() * node_99);
    shared_vars[7] -= 0.010000 * node_99.colwise().sum().transpose();
    shared_vars[8] -= 0.010000 * (node_32.transpose() * node_89);
    shared_vars[9] -= 0.010000 * node_89.colwise().sum().transpose();
    shared_vars[10] -= 0.010000 * (node_36.transpose() * node_79);
    shared_vars[11] -= 0.010000 * node_79.colwise().sum().transpose();
    shared_vars[12] -= 0.010000 * (node_40.transpose() * node_69);
    shared_vars[13] -= 0.010000 * node_69.colwise().sum().transpose();
    shared_vars[14] -= 0.010000 * (node_44.transpose() * node_59);
    shared_vars[15] -= 0.010000 * node_59.colwise().sum().transpose();
    // Write all of the output nodes in correct order
    Eigen::MatrixXf one(1, 1);
    one << 1.0;
    Eigen::MatrixXf node_54(1,1);
    node_54 << (inputs[0].array() * softplus(-node_47, 50).array() + (-inputs[0] + Eigen::MatrixXf::Ones(1000, 784)).array() * softplus(node_47, 50).array()).sum() * 0.001000;
    return {node_54};
}

int main(int argc, char **argv) {
    Eigen::setNbThreads(4);
    std::string name = "mnist_hinton_e";
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
    download_mnist(name);
    float * data_ptr = new float[28*28*num_images]{};
    int* labels_ptr = new int[num_images]{};
    ReadTrainMNIST(name, data_ptr, labels_ptr);
    Eigen::Map<Eigen::MatrixXf> data(data_ptr, num_images, 28*28);
    Eigen::Map<Eigen::VectorXi> l_in(labels_ptr, num_images);

    // Default batch size of 1000
    int batch_size = 1000;
    std::vector<Eigen::MatrixXf> shared_vars;
    std::vector<Eigen::MatrixXf> data_inv;
    std::vector<Eigen::MatrixXf> results;
    // Initialize shared_vars
    int d[9] = {784, 1000, 500, 250, 30, 250, 500, 1000, 784};
    for(int i=1;i<9;i++){
        shared_vars.push_back(Eigen::MatrixXf::Random(d[i-1], d[i]) / 100.0 );
        shared_vars.push_back(Eigen::VectorXf::Zero(d[i]));
    }
    // Number of epochs for burnout, to be discarded
    int burnout = 50;
    // Number of epochs
    int epochs = 100 + burnout;
    Eigen::MatrixXf hv;
    clock_t start;
    for(int i=0;i<epochs;i++){
        if(i == burnout){
            start = clock();
        }
        int ind = i % (num_images / batch_size);
        data_inv.push_back({data.middleRows(ind*batch_size, batch_size)});
        results = eval_func(data_inv, shared_vars);
        hv = results[0];
        std::cout << hv << " (" << data_inv[0].rows() << ", " << data_inv[0].cols() << ")" << std::endl;
        data_inv.clear();
    }
    hv = results[0];
    long int time = (clock() - start);
    std::cout << "Final Value: " << hv(0) << std::endl;
    std::cout << "Mean run time: " << 1000*(double (time))/(CLOCKS_PER_SEC*(epochs - burnout)) << "ms" << std::endl;
}

static size_t my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream)
{
    struct FtpFile *out=(struct FtpFile *)stream;
    if(out && !out->stream) {
        /* open file for writing */
        out->stream=fopen(out->filename, "wb");
        if(!out->stream)
            return -1; /* failure, can't open file to write */
    }
    return fwrite(buffer, size, nmemb, out->stream);
}


inline bool exists (const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

void download_mnist(std::string folder){
    CURL *curl;
    CURLcode res;
    auto data = folder;
    data += kPathSeparator;
    data += "train-images-idx3-ubyte.gz";
    struct FtpFile training_data={
            data.c_str(), /* name to store the file as if successful */
            NULL
    };
    auto labels = folder;
    labels += kPathSeparator;
    labels += "train-labels-idx1-ubyte.gz";
    struct FtpFile training_labels={
            labels.c_str(), /* name to store the file as if successful */
            NULL
    };


    curl_global_init(CURL_GLOBAL_DEFAULT);

    curl = curl_easy_init();
    if(curl) {
        /*
         * You better replace the URL with one that works!
         */
        if(not exists(data.substr(0, data.length()-3))){
            curl_easy_setopt(curl, CURLOPT_URL,
                             "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
            /* Define our callback to get called when there's data to be written */
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, my_fwrite);
            /* Set a pointer to our struct to pass to the callback */
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &training_data);
            res = curl_easy_perform(curl);
            if(CURLE_OK != res) {
                /* we failed */
                fprintf(stderr, "curl told us %d\n", res);
            }
        }
        if(not exists(labels.substr(0, data.length()-3))){
            curl_easy_setopt(curl, CURLOPT_URL,
                             "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
            /* Define our callback to get called when there's data to be written */
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, my_fwrite);
            /* Set a pointer to our struct to pass to the callback */
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &training_labels);
            res = curl_easy_perform(curl);
            if(CURLE_OK != res) {
                /* we failed */
                fprintf(stderr, "curl told us %d\n", res);
            }
        }
    }
    /* always cleanup */
    curl_easy_cleanup(curl);

    if(training_data.stream) {
        fclose(training_data.stream); /* close the local file */
        int res = system(("gzip -d " + data).c_str());
    }

    if(training_labels.stream) {
        fclose(training_labels.stream); /* close the local file */
        int res = system(("gzip -d " + labels).c_str());
    }

    curl_global_cleanup();
}

int ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void ReadTrainMNIST(std::string folder, float* data, int* labels){
    Eigen::setNbThreads(4);
    std::string file_name = folder;
    file_name += kPathSeparator;
    file_name += "train-images-idx3-ubyte";
    std::ifstream file(file_name,std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        std::cout << number_of_images << ", " << rows << ", " << cols << std::endl;
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    data[(r*cols + c)*num_images + i]= ((float)temp) / float(255.0);
                }
            }
        }
    }
    file.close();
    file_name = folder;
    file_name += kPathSeparator;
    file_name += "train-labels-idx1-ubyte";
    file.open(file_name, std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        std::cout << number_of_images << std::endl;
        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            labels[i]= (int)temp;
        }
    }
    file.close();
}


