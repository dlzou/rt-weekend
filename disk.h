#ifndef GROUND_H
#define GROUND_H

#include "hittable.h"
#include "interval.h"
#include "material.h"
#include "vec3.h"

class disk : public hittable {
public:
    __device__ disk(const ray &center_normal, float radius, material *mat)
        : center_normal(center_normal.origin(), unit_vector(center_normal.direction())),
          radius(max(0.0f, radius)), mat(mat) {}

    __device__ disk(const disk &other)
        : center_normal(other.center_normal), radius(other.radius), mat(other.mat->clone()) {}

    __device__ disk &operator=(const disk &other) {
        if (this != &other) {
            delete mat;
            center_normal = other.center_normal;
            radius = other.radius;
            mat = other.mat->clone();
        }
        return *this;
    }

    __device__ ~disk() override { delete mat; }

    __device__ hittable *clone() const override { return new disk(*this); }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        vec3 cr = center_normal.origin() - r.origin();
        float ncr = dot(center_normal.direction(), cr);
        float t = ncr / dot(center_normal.direction(), r.direction());
        if (!ray_t.surrounds(t)) {
            return false;
        }

        rec.t = t;
        rec.p = r.point_at(rec.t);
        if ((rec.p - center_normal.origin()).length() > radius) {
            return false;
        }
        vec3 outward_normal = center_normal.direction();
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }

private:
    ray center_normal;
    float radius;
    material *mat;
};

#endif
