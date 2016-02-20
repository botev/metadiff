//
// Created by alex on 19/01/16.
//

#include "iostream"
#include "math.h"
#include "vector"
#include <omp.h>
#include "mkl.h"
#include "curl/curl.h"
#include <sys/stat.h>
#include <fstream>

const int N = 1000;
const int d[9] = {784, 1000, 500, 250, 30, 250, 500, 1000, 784};
const int size = 1000*d[0];
const int inc = 1;

struct matrix {
    int d1;
    int d2;
    float* storage;

};

std::vector<matrix*> registers;
std::vector<matrix*> shared_vars;
matrix* ones;
matrix* result;


void add_bias_and_tanh(matrix* wx, const matrix* bias){
#pragma omp parallel for
    for(int j=0;j<wx->d2;j++){
        for(int i=0;i<wx->d1;i++){
            wx->storage[i + j*wx->d1] = tanhf(wx->storage[i + j*wx->d1] + bias->storage[wx->d2]);
        }
    }
}

void add_bias(matrix* wx, const matrix* bias){
#pragma omp parallel for
    for(int j=0;j<wx->d2;j++){
        for(int i=0;i<wx->d1;i++){
            wx->storage[i + j*wx->d1] += bias->storage[wx->d2];
        }
    }
}

void entropy(matrix *res, matrix* p, matrix *q){
#pragma omp parallel for
    for(int j=0;j<res->d2;j++){
        for(int i=0;i<res->d1;i++){

            res->storage[i + j*res->d1] += 0.001 * (1 / (1 + expf(q->storage[i + j*res->d1])) - p->storage[i + j*res->d1]);
        }
    }
}

void dowtv(matrix *res, matrix *r){
#pragma omp parallel for
    for(int j=0;j<res->d2;j++){
        for(int i=0;i<res->d1;i++){
            res->storage[i + j*res->d1] *= 1 - r->storage[i + j*res->d1] * r->storage[i + j*res->d1];
        }
    }
}

void update_matrix(matrix* l, matrix* r, matrix* res){
//    std::cout << l->d1 << ", " << l->d2 << std::endl;
//    std::cout << r->d1 << ", " << r->d2 << std::endl;
//    std::cout << res->d1 << ", " << res->d2 << std::endl;
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, res->d1, res->d2, l->d1, -0.01, l->storage, l->d1, r->storage, r->d1, 1, res->storage, res->d1);
}

inline void gemm(matrix* a, matrix* b, matrix* c, float alpha = 1.0,
                 CBLAS_TRANSPOSE at = CblasNoTrans, CBLAS_TRANSPOSE bt = CblasNoTrans){
    const int m = c->d1;
    const int k = at == CblasNoTrans ? a->d2 : a->d1;
    const int n = c->d2;
    cblas_sgemm(CblasColMajor, at, bt, m, n, k, alpha, a->storage, a->d1, b->storage, b->d1, 0, c->storage, c->d1);
}

void final(matrix* res, matrix* node47, matrix* in){
#pragma omp parallel for
    for(int j=0;j<res->d2;j++){
        for(int i=0;i<res->d1;i++){
            res->storage[i + j*res->d1] = in->storage[i + j*res->d1] * log1pf(expf(-node47->storage[i + j*res->d1]))
                                          + (1 - in->storage[i + j*res->d1]) * log1pf(expf(node47->storage[i + j*res->d1]));
        }
    }
}

void initialize_registers_and_shared(){
    // All shared vars
    matrix* temp;
    for(int i=0;i<8;i++){
        temp = new matrix;
        temp->d1 = d[i];
        temp->d2 = d[i+1];
        temp->storage = (float*) malloc(temp->d1*temp->d2*sizeof(float));
        shared_vars.push_back(temp);
        temp = new matrix;
        temp->d1 = 1;
        temp->d2 = d[i+1];
        temp->storage = (float*) malloc(temp->d1*temp->d2*sizeof(float));
        shared_vars.push_back(temp);
    }
    // All registers 0-7
    for(int i=0;i<8; i++){
        temp = new matrix;
        temp->d1 = N;
        temp->d2 = d[i+1];
        temp->storage = (float*) malloc(temp->d1*temp->d2*sizeof(float));
        registers.push_back(temp);
    }
    // 8
    temp = new matrix;
    temp->d1 = N;
    temp->d2 = d[8];
    temp->storage = (float*) malloc(temp->d1*temp->d2*sizeof(float));
    registers.push_back(temp);
    // 9-15
    for(int i=0;i<7;i++){
        temp = new matrix;
        temp->d1 = registers[6-i]->d1;
        temp->d2 = registers[6-i]->d2;
        temp->storage = (float*) malloc(temp->d1*temp->d2*sizeof(float));
        registers.push_back(temp);
    }
    ones = new matrix;
    ones->d1 = N;
    ones->d2 = 1;
    ones->storage = (float*) malloc(N*sizeof(float));
    result = new matrix;
    result->d1 = N;
    result->d2 = d[8];
    result->storage = (float*) malloc(N*d[8]*sizeof(float));

}
float eval_func(std::vector<matrix*>& inputs){
    // 0: 20, 1:24, 2:28, 3:32, 4:36, 5:40, 6:44, 7:47, 8:59
    gemm(inputs[0], shared_vars[0], registers[0]);
    add_bias_and_tanh(registers[0], shared_vars[1]);
    for(int i=0;i<6;i++){
        gemm(registers[i], shared_vars[2*i+2], registers[i+1]);
        add_bias_and_tanh(registers[i+1], shared_vars[2*i+3]);
//        std::cout << i << std::endl;
    }
    gemm(registers[6], shared_vars[14], registers[7]);
    add_bias(registers[7], shared_vars[15]);
    entropy(registers[8], inputs[0], registers[7]);

    // 9: 69, 10: 79, 11:89, 12:99, 13:109, 14:119, 15:129
    matrix* prev = registers[8];
    for(int i=0;i<7;i++){
        gemm(prev, shared_vars[14-2*i], registers[9+i], 1.0, CblasNoTrans, CblasTrans);
        dowtv(registers[9+i], registers[6-i]);
        prev = registers[9+i];
    }
//    std::cout  << "K" << std::endl;
    // Updates
    update_matrix(inputs[0], registers[15], shared_vars[0]);
//    std::cout << "M" << std::endl;
    update_matrix(ones, registers[15], shared_vars[1]);
    for(int i=0;i<7;i++){
        update_matrix(registers[i], registers[14-i], shared_vars[2*i+2]);
        update_matrix(ones, registers[14-i], shared_vars[2*i+3]);
    }
    final(result, registers[7], inputs[0]);
    return sasum(&size, result->storage, &inc);
}

void ReadTrainMNIST(std::string folder, float* data, int* labels);
void download_mnist(std::string folder);


const int num_images = 60000;
const int rows = 28;
const int cols = 28;

int main(int argc, char **argv) {
    // Download and load MNIST
    std::string name = "mnist_hinton";
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
    download_mnist(name);
    float * data_ptr = new float[28*28*num_images]{};
    int* labels_ptr = new int[num_images]{};
    ReadTrainMNIST(name, data_ptr, labels_ptr);

    omp_set_num_threads(4);
    initialize_registers_and_shared();
    matrix* inputs = new matrix;
    inputs->d1 = N;
    inputs->d2 = d[0];

//    std::cout << "begin" << std::endl;
    std::vector<matrix*> ins = {inputs};
    clock_t time;
    clock_t start = clock();
    int epochs = 100;
    int batch_size = 1000;
    for(int i=0;i<100;i++){
        int ind = i % (num_images / batch_size);
        ins[0]->storage = data_ptr+28*28*ind;
        float r = eval_func(ins);
        std::cout << r << std::endl;
    }
    time = (clock() - start);
    std::cout << "Mean run time: " << 1000*((double) (time))/((double) (CLOCKS_PER_SEC*(100))) << "ms" << std::endl;
}

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

