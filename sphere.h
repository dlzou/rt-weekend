#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "interval.h"
#include "vec3.h"

class sphere : public hittable {
public:
    __device__ sphere(const point3 &center, float radius, material *mat)
        : center(center), radius(max(0.0f, radius)), mat(mat) {}

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        vec3 oc = center - r.origin();
        float a = r.direction().length_squared();
        float h = dot(r.direction(), oc); // h = -b/2
        float c = oc.length_squared() - radius * radius;

        float discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.point_at(rec.t);
        rec.normal = (rec.p - center) / radius;
        rec.mat = mat;

        return true;
    }

private:
    point3 center;
    float radius;
    material *mat;
};

#endif
