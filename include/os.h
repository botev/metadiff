//
// Created by alex on 30/05/16.
//

#ifndef METADIFF_OS_H
#define METADIFF_OS_H

#include "curl/curl.h"
#include <sys/stat.h>
#include "fstream"

namespace metadiff{
    namespace os{
        const char kPathSeparator =
#ifdef _WIN32
                '\\';
#else
                '/';
#endif

        /** Checks if a the file specified by the path exists on the file system
         * TODO - make this cross-platform */
        bool exists(std::string path){
            struct stat buffer;
            return (stat (path.c_str(), &buffer) == 0);
        }

        /** Returns false if the path does not exists or if it is not a directory
         * TODO - make this cross-platform */
        bool is_dir(std::string path){
            struct stat info;
            if(stat( path.c_str(), &info ) != 0)
                return false;
            else if(info.st_mode & S_IFDIR)
                return true;
            else
                return false;
        }

        /** Helper function to create directories
         * If check is true the directory is created only if it does not exists
         * TODO - make this cross-platform */
        void create_dir(std::string path, bool check = false) {
            if((not check) or (not is_dir(path))) {
                const int dir_err = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                if (-1 == dir_err) {
                    std::cout << "Error creating directory " << path << std::endl;
                }
            }
        }

        /** Function to create a temporary directory and return its path
         * TODO - make this cross-platform */
        std::string make_temp_dir() {
            std::string path = std::tmpnam(nullptr);
            return path;
        };

        /** Joins os paths **/
        std::string join_paths(std::vector<std::string> paths){
            if(paths.size() == 0){
                return "";
            } else{
                std::string path = paths[0];
                for(auto i = 1; i < paths.size(); ++i){
                    if(path.back() != kPathSeparator){
                        path += kPathSeparator;
                    }
                    if(paths[i].front() == kPathSeparator){
                        path += paths[i].substr(1, paths[i].size());
                    } else {
                        path += paths[i];
                    }
                }
                if(path.back() == kPathSeparator){
                    return path.substr(0, path.size() - 1);
                } else {
                    return path;
                }
            }
        }

        /** Joins os paths **/
        std::string join_paths(std::string path1, std::string path2){
            return join_paths({path1, path2});
        }

        /** The name of the environmental variable expected for a http proxy */
        const std::string HTTP_PROXY = "HTTP_PROXY";

        /** Helper for writing to files */
        struct FtpFile {
            const char *filename;
            FILE *stream;
        };

        /** Helper writes stream to files */
        static size_t write_steram_to_file(void *buffer, size_t size, size_t nmemb, void *stream) {
            struct FtpFile *out=(struct FtpFile *)stream;
            if(out && !out->stream) {
                out->stream=fopen(out->filename, "wb");
                if(!out->stream)
                    return -1; /* failure, can't open file to write */
            }
            return fwrite(buffer, size, nmemb, out->stream);
        }

        /** A helper function to download a file from a url */
        void download_file(std::string url, std::string local_path){
            CURL *curl;
            CURLcode res;
            curl_global_init(CURL_GLOBAL_DEFAULT);
            curl = curl_easy_init();
            struct FtpFile data={local_path.c_str(), NULL};
            if(curl) {
                if(std::getenv(HTTP_PROXY.c_str())){
                    curl_easy_setopt(curl, CURLOPT_PROXY, std::getenv(HTTP_PROXY.c_str()));
                }
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &write_steram_to_file);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
                res = curl_easy_perform(curl);
                if(CURLE_OK != res) {
                    fprintf(stderr, "curl told us %d\n", res);
                }
            }
            /* always cleanup */
            curl_easy_cleanup(curl);
            if(data.stream) {
                fclose(data.stream);
            }
            curl_global_cleanup();
        }

        /** Returns the size of a file in bytes
         * TODO Check if this is truly correct way to do this */
        long long file_size(std::string file_name) {
            struct stat st;
            if (stat(file_name.c_str(), &st) == -1)
                return 0;
            return ((long long)st.st_size);
        }

        /** Definitely not cross platform, but for now will do */
        int unpack_gz(std::string gz_path){
            return system(("gzip -d -f " + gz_path).c_str());
        }
    }
}

#endif //METADIFF_OS_H
