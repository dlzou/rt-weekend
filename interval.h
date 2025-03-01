#ifndef INTERVAL_H
#define INTERVAL_H

#include <cuda_runtime.h>

class interval {
public:
    float min, max;

    __host__ __device__ interval() : min(INFINITY), max(-INFINITY) {}
    __host__ __device__ interval(float _min, float _max) : min(_min), max(_max) {}

    __host__ __device__ float size() const { return max - min; }

    __host__ __device__ bool contains(float x) const { return min <= x && x <= max; }

    __host__ __device__ bool surrounds(float x) const { return min < x && x < max; }

    __host__ __device__ float clamp(float x) const {
        if (x <= min)
            return min;
        if (x >= max)
            return max;
        return x;
    }

    static const interval empty, universe;
};

const interval interval::empty = interval(INFINITY, -INFINITY);
const interval interval::universe = interval(-INFINITY, INFINITY);

#endif
