#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

// Constants

const float pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

#include "color.h"
#include "ray.h"
#include "vec3.h"

#endif
