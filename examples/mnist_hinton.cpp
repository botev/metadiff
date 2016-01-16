
#include <arrayfire.h>
#include <sys/stat.h>
#include "metadiff.h"
#include "curl/curl.h"

namespace md = metadiff;
namespace sym = metadiff::symbolic;

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

const int num_images = 60000;
const int rows = 28;
const int cols = 28;

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

int main(int argc, char **argv)
{
    // Download and load MNIST
    std::string name = "mnist_hinton";
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
    download_mnist(name);
    float * data_ptr = new float[28*28*num_images]{};
    int* labels_ptr = new int[num_images]{};
    ReadTrainMNIST(name, data_ptr, labels_ptr);
//    // Default to CPU
//    af_backend backend = AF_BACKEND_CPU;
//    // Default batch size of 1000
//    int batch_size = 1000;
//    if(argc > 3){
//        std::cerr << "Expecting two optional arguments - backend and batch size" << std::endl;
//        exit(1);
//    }
//    if(argc > 1){
//        std::string cpu = "cpu";
//        std::string opencl = "opencl";
//        std::string cuda = "cuda";
//        if(cpu.compare(argv[1]) == 0){
//            backend = AF_BACKEND_CPU;
//        } else if(opencl.compare(argv[1]) == 0) {
//            backend = AF_BACKEND_OPENCL;
//        } else if(cuda.compare(argv[1]) == 0) {
//            backend = AF_BACKEND_CUDA;
//        } else {
//            std::cout << (argv[1] == "opencl") << std::endl;
//            std::cerr << "The first argument should be one of 'cpu', 'opencl' and 'gpu' - " << argv[1] << std::endl;
//            exit(1);
//        }
//    }
//    if(argc > 2){
//        std::istringstream ss(argv[2]);
//        if(!(ss >> batch_size)) {
//            std::cerr << "Invalid number " << argv[2] << '\n';
//        }
//    }
//    af::setBackend(backend);
//
//    // Transfer data to Arrayfire
//    af::array data(num_images, 28*28, data_ptr, afHost);
//    af::array l_in(num_images, labels_ptr, afHost);
//
////    int ind = 2;
////    af_print(af::moddims(data(ind, af::span)>0.5, 28, 28));
////    std::cout << labels_ptr[ind] << std::endl;
////    for(int i=0;i<28;i++){
////        for(int j=0;j<28;j++){
////            std::cout << (data_ptr[(i*28+j)*num_images + ind] > 0.5);
////        }
////        std::cout << std::endl;
////    }
//    // Create graph
//    auto graph = md::create_graph();
//    graph->name = name;
//
//    graph->broadcast = md::ad_implicit_broadcast::WARN;
//    // Batch size
//    auto n = graph->get_new_symbolic_integer(); // a
//    // Real batch size
//    // Architecture
//    int d[9] = {784, 1000, 500, 250, 30, 250, 500, 1000, 784};
//    // Input data
//    auto data_in = graph->matrix(md::FLOAT, {n, d[0]}, "Input");
//    // Parameters
//    std::vector<md::Node> params;
//    for(int i=1;i<9;i++){
//        params.push_back(graph->shared_var(af::randn(d[i-1], d[i]) / 100.0, "W_" + std::to_string(i)));
//        params.push_back(graph->shared_var(af::constant(0.0, 1, d[i]), "b_" + std::to_string(i)));
//    }
//    // Input Layer
//    auto h = md::relu(md::dot(data_in, params[0]) + params[1]);
//    // All layers except one
//    for(int i=1;i<7;i++){
//        h = md::relu(md::dot(h, params[2*i]) + params[2*i+1]);
//    }
//    // Calculate only logits here
//    h = md::dot(h, params[14]) + params[15];
//    // Loss
//    auto error = md::binary_cross_entropy_logit(data_in, h);
//    //o Mean loss
//    auto loss = error->sum() * graph->value(1.0 / float(batch_size));
//    // Get grads
//    auto grads = graph->gradient(loss, params);
//    // Learning rate
//    auto learning_rate = graph->constant_node(af::constant(0.01, 1));
//
//    // Set up sgd
//    md::Updates updates;
//    for(int i=0;i<params.size();i++){
//        updates.push_back(std::pair<md::Node, md::Node>(params[i], params[i] - learning_rate * grads[i]));
//    }
//
//    name += kPathSeparator + name;
//    // Print to file
//    md::dagre::dagre_to_file(name + ".html", graph, {loss}, updates);
//
//    // Create backend and compile function
//    md::ArrayfireBackend md_backend("/opt/arrayfire-3/include", "/opt/arrayfire-3/lib");
//    auto train = md_backend.compile_function(name, graph, {data_in}, {loss}, updates);
//
//    // Run function
//    long long time = 0;
//    long long min_time = 1000 * 60 * 60 * 24;
//    long long max_time = 0;
//
//    // Number of epochs for burnout, to be discarded
//    int burnout = 20;
//    // Number of epochs
//    int epochs = 100 + burnout;
//    float *hv;
//    std::vector<af::array> data_inv;
//    for(int i=0;i<epochs;i++){
//        int ind = i % (num_images / batch_size);
//        // Input data
////        std::cout << data_inv[0].dims() << std::cout;
//        auto ptr = data_ptr + ind * batch_size * 28 * 28;
//        clock_t start = clock();
//        data_inv = {af::array(batch_size,  28*28,  ptr,  afHost)};
//        auto result = train.eval(data_inv);
//        hv = result[0].host<float>();
//        clock_t end = clock();
////        std::cout << "Value: " << hv[0] << std::endl;
////        std::cout << "Elapsed time: " << 1000*(double(end - start))/CLOCKS_PER_SEC << "ms" << std::endl;
//        if(i >= burnout) {
//            auto run_time = (end - start);
//            if(run_time < min_time){
//                min_time = run_time;
//            }
//            if(run_time > max_time){
//                max_time = run_time;
//            }
//            time += run_time;
//        }
//    }
//    md_backend.close();
//    std::cout << "Final Value: " << hv[0] << std::endl;
//    std::cout << "Mean run time: " << 1000*(double (time))/(CLOCKS_PER_SEC*(epochs - burnout)) << "ms" << std::endl;
//    std::cout << "Max run time: " << 1000*(double (max_time))/CLOCKS_PER_SEC << "ms" << std::endl;
//    std::cout << "Min run time: " << 1000*(double (min_time))/CLOCKS_PER_SEC << "ms" << std::endl;
//    return 0;
}