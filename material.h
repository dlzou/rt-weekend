#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"

class material {
public:
    __device__ virtual ~material() {};
    __device__ virtual material *clone() const = 0;

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
                                    ray &scattered, curandState *rs, bool debug) const {
        return false;
    }
};

class lambertian : public material {
public:
    __device__ lambertian(const color &albedo) : albedo(albedo) {}

    __device__ lambertian *clone() const override {
        return new lambertian(*this);
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
                            ray &scattered, curandState *rs, bool debug = false) const override {
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

    __device__ metal *clone() const override {
        return new metal(*this);
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
                            ray &scattered, curandState *rs, bool debug = false) const override {
        vec3 direction = reflect(r_in.direction(), rec.normal);
        direction = unit_vector(direction) + (fuzz * random_unit_vector(rs));
        scattered = ray(rec.p, direction);
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0;
    }

private:
    color albedo;
    float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ dielectric *clone() const override {
        return new dielectric(*this);
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
                            ray &scattered, curandState *rs, bool debug = false) const override {
        attenuation = color(1.0, 1.0, 1.0);
        float eta_ratio = rec.front_face ? (1.0 / refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = eta_ratio * sin_theta > 1.0;
        vec3 direction;
        if (cannot_refract || reflectance(cos_theta, eta_ratio) > curand_uniform(rs))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, eta_ratio, debug);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    // Refractive index of material when enclosed by vacuum
    float refraction_index;
    
    __device__ static float reflectance(float cos, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cos), 5);
    }
};

#endif