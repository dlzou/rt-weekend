#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"

class material {
public:
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
                                    ray &scattered, curandState *rs) const {
        return false;
    }
};

class lambertian : public material {
public:
    __device__ lambertian(const color &albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
                            ray &scattered, curandState *rs) const override {
        vec3 direction = rec.normal + random_unit_vector(rs);
        if (direction.near_zero())
            direction = rec.normal;

        scattered = ray(rec.p, direction);
        attenuation = albedo;
        return true;
    }

private:
    color albedo;
};

class metal : public material {
public:
    __device__ metal(const color &albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
                            ray &scattered, curandState *rs) const override {
        vec3 direction = reflect(r_in.direction(), rec.normal);
        direction = unit_vector(direction) +  (fuzz * random_unit_vector(rs));
        scattered = ray(rec.p, direction);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

private:
    color albedo;
    float fuzz;
};

#endif
