//
// Created by alex on 22/01/16.
//

#ifndef METADIFF_MNIST_H
#define METADIFF_MNIST_H
#include "curl/curl.h"

namespace datasets {
    const int MNIST_NUM_IMAGES = 60000;
    const int MNIST_NUM_ROWS = 28;
    const int MNIST_NUM_COLS = 28;
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
//            std::cout << number_of_images << ", " << MNIST_NUM_ROWS << ", " << MNIST_NUM_COLS << std::endl;
            for(int i=0;i<number_of_images;++i)
            {
                for(int r=0;r<n_rows;++r)
                {
                    for(int c=0;c<n_cols;++c)
                    {
                        unsigned char temp=0;
                        file.read((char*)&temp,sizeof(temp));
                        data[(n_rows*r)+c + i*MNIST_NUM_COLS*MNIST_NUM_ROWS]= ((float)temp) / float(255.0);
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
//            std::cout << number_of_images << std::endl;
            for(int i=0;i<number_of_images;++i)
            {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                labels[i]= (int)temp;
            }
        }
        file.close();
    }
}

#endif //METADIFF_MNIST_H
