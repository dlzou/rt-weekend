#ifndef HITTABLE_H
#define HITTABLE_H

#include "interval.h"

class hit_record {
public:
    point3 p;
    vec3 normal;
    float t;
    bool front_face; // Ray hits face from the front, i.e. outside of object

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        front_face = dot(r.direction(), outward_normal) < 0; // Ray hits face from outside
        normal = front_face ? outward_normal : -outward_normal; // Normal always points against ray
    }
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};

#endif
