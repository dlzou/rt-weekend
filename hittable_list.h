#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include "interval.h"

#include <vector>

class hittable_list : public hittable {
public:
    hittable **objects;
    int size;

    __device__ hittable_list() {}
    __device__ hittable_list(hittable **objects, int size) : objects(objects), size(size) {}
    
    __device__ hittable_list(const hittable_list &other) {
        size = other.size;
        objects = new hittable*[size];
        for (int i = 0; i < size; i++) {
            objects[i] = other.objects[i]->clone();
        }
    } 
    
    __device__ hittable_list &operator=(const hittable_list &other) {
        if (this != &other) {
            for (int i = 0; i < size; i++) {
                delete objects[i];
            }
            delete[] objects;
            size = other.size;
            objects = new hittable*[size];
            for (int i = 0; i < size; i++) {
                objects[i] = other.objects[i]->clone();
            }
        }
        return *this;
    }

    __device__ ~hittable_list() override {
        for (int i = 0; i < size; i++) {
            delete objects[i];
        }
        delete[] objects;
    }
    
    __device__ hittable_list *clone() const override {
        return new hittable_list(*this);
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < size; i++) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif