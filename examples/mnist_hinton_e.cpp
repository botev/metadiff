#define EIGEN_USE_MKL_ALL

#include "Eigen/Core"
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

inline Eigen::ArrayXXf softplus(Eigen::Ref<Eigen::ArrayXXf> input, int threshold) {
    return (input > threshold).select(input, (input.exp()+1).log());
}

int main(int argc, char **argv)
{
    // Download and load MNIST
    std::string name = "mnist_hinton_e";
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH);
    download_mnist(name);
    float * data_ptr = new float[28*28*num_images]{};
    int* labels_ptr = new int[num_images]{};
    ReadTrainMNIST(name, data_ptr, labels_ptr);

    // Default batch size of 1000
    int batch_size = 1000;
    // Default period
    int period = 1;
    if(argc > 3){
        std::cerr << "Expecting two optional arguments - backend and batch size" << std::endl;
        exit(1);
    }
    if(argc > 1){
        std::istringstream ss(argv[1]);
        if(!(ss >> batch_size)) {
            std::cerr << "Invalid number " << argv[1] << '\n';
        }
    }
    if(argc > 2){
        std::istringstream ss(argv[2]);
        if(!(ss >> period)) {
            std::cerr << "Invalid number " << argv[2] << '\n';
        }
    }
    // Transfer data to Arrayfire
    Eigen::Map<Eigen::MatrixXf> data(data_ptr, num_images, 28*28);
    Eigen::Map<Eigen::VectorXi> l_in(labels_ptr, num_images);
    // Create graph
    md::Graph graph = md::create_graph();
    graph->name = name;

    graph->broadcast = md::ad_implicit_broadcast::WARN;
    // Batch size
    auto n = graph->get_new_symbolic_integer(); // a
    // Real batch size
    // Architecture
    int d[9] = {784, 1000, 500, 250, 30, 250, 500, 1000, 784};
    // Input data
    auto test = graph->constant_value(20);
    md::NodeVec inputs = {graph->matrix(md::FLOAT, {n, d[0]}, "Input")};
    // Parameters
    std::vector<md::Node> params;
    for(int i=1;i<9;i++){
        params.push_back(graph->shared_var(Eigen::ArrayXXf::Random(d[i-1], d[i]) / 100.0, "W_" + std::to_string(i)));
        params.push_back(graph->shared_var(Eigen::ArrayXXf::Zero(1, d[i]), "b_" + std::to_string(i)));
    }
    // Input Layer
    auto h = md::tanh(md::dot(inputs[0], params[0]) + params[1]);
    // All layers except one
    for(int i=1;i<7;i++){
        h = md::tanh(md::dot(h, params[2*i]) + params[2*i+1]);
    }
    // Calculate only logits here
    h = md::dot(h, params[14]) + params[15];
    // Loss
    auto error = md::binary_cross_entropy_logit(inputs[0], h);
    // Mean loss
    md::NodeVec loss = {error.sum() * graph->constant_value(1.0 / float(batch_size))};
    // Get grads
    auto grads = graph->gradient(loss[0], params);
    // Learning rate
    auto learning_rate = graph->constant_value(0.01);
    // Set up sgd
    md::Updates updates;
    for(int i=0;i<params.size();i++){
        updates.push_back(std::pair<md::Node, md::Node>(params[i], params[i] - learning_rate * grads[i]));
    }
    name += kPathSeparator + name;
    // Print to file
    md::dagre::dagre_to_file(name + ".html", graph, loss, updates);
    // Optimize
    md::NodeVec new_inputs;
    md::NodeVec new_loss;
    md::Updates new_updates;
    md::Graph optimized =  graph->optimize(loss, updates, inputs,
                                           new_loss, new_updates, new_inputs);
    std::cout << "Original:" << graph->nodes.size() << std::endl;
    std::cout << "Optimized:" << optimized->nodes.size() << std::endl;
    md::dagre::dagre_to_file(name + "_optim.html", optimized, new_loss, new_updates);
    // Create backend and compile function
    md::EigenBackend md_backend = md::EigenBackend();
    md_backend.mkl = true;
//    auto train_org = md_backend.compile_function(name, graph, inputs, loss, updates);
    auto train_optim = md_backend.compile_function(name + "_optim", optimized, new_inputs, new_loss, new_updates);

    // Run function
    long long time = 0;

    // Number of epochs for burnout, to be discarded
    int burnout = 50;
    // Number of epochs
    int epochs = 100;
    Eigen::MatrixXf hv;
    clock_t start;
    std::vector<Eigen::ArrayXXf> result;
    std::vector<Eigen::ArrayXXf> data_inv;
    for(int i=0;i<epochs + burnout;i++){
        if(i == burnout){
             start = clock();
        }
        int ind = i % (num_images / batch_size);
        data_inv = {data.middleRows(ind*batch_size, batch_size)};
        result = train_optim.eval(data_inv);
        if(i >= burnout and i % period == 0) {
            hv = result[0];
        }
    }
    hv = result[0];
    time = (clock() - start);
    md_backend.close();
    std::cout << "Final Value: " << hv(0) << std::endl;
    std::cout << "Mean run time: " << (double (1000*time))/double(CLOCKS_PER_SEC*(epochs)) << "ms" << std::endl;
    return 0;
}
