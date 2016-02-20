
#define SPDLOG_DEBUG_ON
#include <iostream>
#include "algorithm"
#include "spdlog.h"

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


int main(int, char* [])
{
    namespace spd = spdlog;
    try
    {
        auto dist_sink = std::make_shared<dist_sink_st>();
        auto console = std::make_shared<spdlog::logger>("my_logger", dist_sink);
        auto console2 = std::make_shared<spdlog::logger>("another_logger", dist_sink);

        auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_st>();
        dist_sink->add_sink(console_sink);

        //Create console, multithreaded logger
//        auto console = spd::stdout_logger_mt("console");
        console->set_level(spd::level::info); // Set specific logger's log level
        console->info("Welcome to spdlog!") ;
        console->info("An info message example {}..", 1);
        console->info() << "Streams are supported too  " << 1;

        //Formatting examples
        console->info("Easy padding in numbers like {:08d}", 12);
        console->info("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
        console->info("Support for floats {:03.2f}", 1.23456);
        console->info("Positional args are {1} {0}..", "too", "supported");

        console2->info("DADAD");
        auto a = spdlog::get("console22");
        if(not a){
            a =  spd::stdout_logger_mt("console22");
        }
        spdlog::get("console22")->info("2222");
        a->info("2222");
        console->info("{:<30}", "left aligned");
        console->info("{:>30}", "right aligned");
        console->info("{:^30}", "centered");

        console->debug("{:<30}", "left aligned");
        SPDLOG_DEBUG(console, "{:<30}", "left aligned 2")
        SPDLOG_TRACE(console, "{:<30}", "left aligned 2")

        //
        // Runtime log levels
        //
        spd::set_level(spd::level::info); //Set global log level to info
        console->debug("This message shold not be displayed!");
        console->set_level(spd::level::debug); // Set specific logger's log level
        console->debug("Now it should..");

        //
        // Create a file rotating logger with 5mb size max and 3 rotated files
        //
        auto file_logger = spd::rotating_logger_mt("file_logger", "logs/mylogfile", 1048576 * 5, 3);
        for(int i = 0; i < 10; ++i)
            file_logger->info("{} * {} equals {:>10}", i, i, i*i);

        //
        // Create a daily logger - a new file is created every day on 2:30am
        //
        auto daily_logger = spd::daily_logger_mt("daily_logger", "logs/daily", 2, 30);

        //
        // Customize msg format for all messages
        //
        spd::set_pattern("*** [%H:%M:%S %z] [thread %t] %v ***");
        file_logger->info("This is another message with custom format");

        spd::get("console")->info("loggers can be retrieved from a global registry using the spdlog::get(logger_name) function");

        //
        // Compile time debug or trace macros.
        // Enabled #ifdef SPDLOG_DEBUG_ON or #ifdef SPDLOG_TRACE_ON
        //
        SPDLOG_TRACE(console, "Enabled only #ifdef SPDLOG_TRACE_ON..{} ,{}", 1, 3.23);
        SPDLOG_DEBUG(console, "Enabled only #ifdef SPDLOG_DEBUG_ON.. {} ,{}", 1, 3.23);

        //
        // Asynchronous logging is very fast..
        // Just call spdlog::set_async_mode(q_size) and all created loggers from now on will be asynchronous..
        //
        size_t q_size = 1048576; //queue size must be power of 2
        spdlog::set_async_mode(q_size);
        auto async_file= spd::daily_logger_st("async_file_logger", "logs/async_log.txt");
        async_file->info() << "This is async log.." << "Should be very fast!";

        //
        // syslog example. linux only..
        //
#ifdef __linux__
        std::string ident = "spdlog-example";
        auto syslog_logger = spd::syslog_logger("syslog", ident, LOG_PID);
        syslog_logger->warn("This is warning that will end up in syslog. This is Linux only!");
#endif
    }
    catch (const spd::spdlog_ex& ex)
    {
        std::cout << "Log failed: " << ex.what() << std::endl;
    }
}


// Example of user defined class with operator<<
class some_class {};
std::ostream& operator<<(std::ostream& os, const some_class& c) { return os << "some_class"; }

void custom_class_example()
{
    some_class c;
    spdlog::get("console")->info("custom class with operator<<: {}..", c);
    spdlog::get("console")->info() << "custom class with operator<<: " << c << "..";
}