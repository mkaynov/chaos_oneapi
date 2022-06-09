
#ifndef CHAOS_RAIITIMER_H
#define CHAOS_RAIITIMER_H

#include <functional>
#include <chrono>


class RaiiTimer {

    std::chrono::high_resolution_clock::time_point t0;
    std::function<void(int)> cb;
    uint32_t* clock;

public:
    RaiiTimer(std::function<void(long long)> callback) : t0(std::chrono::high_resolution_clock::now()), cb(callback) {}
    RaiiTimer(uint32_t* clock_) : t0(std::chrono::high_resolution_clock::now()), clock(clock_) {};

    ~RaiiTimer() {
        auto t1 = std::chrono::high_resolution_clock::now();
        uint32_t nanos = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count());
        if (clock) {
            *clock = nanos;
        } else {
            cb(nanos);
        }
    }
};


#endif //CHAOS_RAIITIMER_H
