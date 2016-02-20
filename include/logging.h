//
// Created by alex on 20/02/16.
//

#ifndef METADIFF_LOGGING_H
#define METADIFF_LOGGING_H

#include "spdlog.h"
namespace metadiff{
    template<class Mutex>
    class dist_sink: public spdlog::sinks::base_sink<Mutex>
    {
    public:
        explicit dist_sink() :_sinks() {}
        dist_sink(const dist_sink&) = delete;
        dist_sink& operator=(const dist_sink&) = delete;
        virtual ~dist_sink() = default;

    protected:
        void _sink_it(const spdlog::details::log_msg& msg) override
        {
            for (auto iter = _sinks.begin(); iter != _sinks.end(); iter++)
                (*iter)->log(msg);
        }

        std::vector<std::shared_ptr<spdlog::sinks::sink>> _sinks;

    public:
        void flush() override
        {
            std::lock_guard<Mutex> lock(spdlog::sinks::base_sink<Mutex>::_mutex);
            for (auto iter = _sinks.begin(); iter != _sinks.end(); iter++)
                (*iter)->flush();
        }

        void add_sink(std::shared_ptr<spdlog::sinks::sink> sink)
        {
            std::lock_guard<Mutex> lock(spdlog::sinks::base_sink<Mutex>::_mutex);
            if (sink &&
                _sinks.end() == std::find(_sinks.begin(), _sinks.end(), sink))
            {
                _sinks.push_back(sink);
            }
        }

        void remove_sink(std::shared_ptr<spdlog::sinks::sink> sink)
        {
            std::lock_guard<Mutex> lock(spdlog::sinks::base_sink<Mutex>::_mutex);
            auto pos = std::find(_sinks.begin(), _sinks.end(), sink);
            if (pos != _sinks.end())
            {
                _sinks.erase(pos);
            }
        }
    };

    typedef dist_sink<std::mutex> dist_sink_mt;
    typedef dist_sink<spdlog::details::null_mutex> dist_sink_st;

    static const std::shared_ptr<dist_sink_st> metadiff_sink = std::make_shared<dist_sink_st>();

    std::shared_ptr<spdlog::logger> logger(std::string name){
        std::shared_ptr<spdlog::logger> ptr = spdlog::get(name);
        if(not ptr){
            ptr = std::make_shared<spdlog::logger>(name, metadiff_sink);
            spdlog::register_logger(ptr);
            ptr->set_level(spdlog::level::trace);
            ptr->set_pattern("[%H:%M:%S][%l][%n::%v");
        }
        return ptr;
    }

}
#endif //METADIFF_LOGGING_H
