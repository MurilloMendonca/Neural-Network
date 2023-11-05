#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <stdexcept>

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

	int threads;

public:
    ThreadPool(size_t);
    auto enqueue(std::function<void()> f) -> std::future<void>;
    ~ThreadPool();
	int threadCount();

};
