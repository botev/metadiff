//
// Created by alex on 20/02/16.
//

#ifndef METADIFF_LOGGING_H
#define METADIFF_LOGGING_H

#include "spdlog/spdlog.h"
#include "spdlog/sinks/dist_sink.h"
namespace metadiff{
    namespace logging {
        static auto metadiff_sink = std::make_shared<spdlog::sinks::dist_sink_st>();

        std::shared_ptr<spdlog::logger> logger(std::string name) {
            std::shared_ptr<spdlog::logger> ptr = spdlog::get(name);
            if (not ptr) {
                ptr = std::make_shared<spdlog::logger>(name, metadiff_sink);
                spdlog::register_logger(ptr);
                ptr->set_level(spdlog::level::trace);
                ptr->set_pattern("[%H:%M:%S][%l][%n] %v");
            }
            return ptr;
        }
    }
}
#endif //METADIFF_LOGGING_H
