#include "rt_weekend.h"

#include "camera.h"
#include "disk.h"
#include "hittable.h"
#include "hittable_list.h"
#include "interval.h"
#include "material.h"
#include "sphere.h"

#include <curand_kernel.h>

#include <chrono>
#include <fstream>
#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file,
                int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":"
                  << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void create_world(hittable **world, int n_objects, curandState *rs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rs);

        // material *mat_ground = new lambertian(color(0.8, 0.8, 0.8));
        material *mat_ground = new metal(color(0.5, 0.5, 0.5), 0.3);
        material *mat1 = new lambertian(color(0.1, 0.2, 0.5));
        material *mat2 = new dielectric(1.5);
        material *mat3 = new metal(color(0.8, 0.6, 0.2), 0);
        
        hittable **objects = new hittable*[n_objects];

        // These dereferenced assignments to dynamic objects require double pointers.
        objects[0] = new disk(ray(point3(0, 0, 0), vec3(0, 1, 0)), 10, mat_ground);
        objects[1] = new sphere(point3(-3.5, 1, -0.8), 1, mat1);
        objects[2] = new sphere(point3(-0.5, 1, 2), 1, mat2);
        objects[3] = new sphere(point3(2.5, 1, 1.5), 1, mat3);
        
        for (int i = 4; i < n_objects; i++) {
            float choose_mat = curand_uniform(rs);
            point3 center(curand_uniform(rs)*10-5, 0.2, curand_uniform(rs)*10-5);

            if (choose_mat < 1.0/3.0) {
                color albedo = color::random(rs);
                material *mat = new lambertian(albedo);
                objects[i] = new sphere(center, 0.2, mat);
            } else if (choose_mat < 2.0/3.0) {
                color albedo = color::random(0.5, 1, rs);
                float fuzz = curand_uniform(rs) / 2;
                material *mat = new metal(albedo, fuzz);
                objects[i] = new sphere(center, 0.2, mat);
            } else {
                material *mat = new dielectric(1.5);
                objects[i] = new sphere(center, 0.2, mat);
            }
        }

        *world = new hittable_list(objects, n_objects);
    }
}

__global__ void free_world(hittable **world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *world;
    }
}

__global__ void init_camera(camera **cam, int iw, int ih, float vfov, point3 look_from,
                            point3 look_at, vec3 vup, float defocus_angle, float focus_dist) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *cam = new camera(iw, ih, vfov, look_from, look_at, vup, defocus_angle, focus_dist);
    }
}

__global__ void free_camera(camera **cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *cam;
    }
}

__global__ void init_render(curandState *rand_state, int iw, int ih) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= iw) || (j >= ih))
        return;

    int pixel_index = j * iw + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__device__ color ray_color(const ray &r, const hittable **world, curandState *rs) {
    ray cur_ray = r;
    color cur_attenuation = color(1.0, 1.0, 1.0);
    bool debug = false;
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, interval(0.001, INFINITY), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(cur_ray, rec, attenuation, scattered, rs, debug)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return color(0, 0, 0);
            }
        } else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float a = 0.5 * (unit_direction.y() + 1.0);
            color bg = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
            return cur_attenuation * bg;
        }
    }
}

__global__ void render(color *fb, int iw, int ih, int n_samples, camera **cam, hittable **world,
                       curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= iw) || (j >= ih))
        return;

    int pixel_index = j * iw + i;
    curandState rs = rand_state[pixel_index];
    color c(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < n_samples; s++) {
        ray r = (*cam)->get_ray(i, j, &rs);
        c += ray_color(r, const_cast<const hittable **>(world), &rs);
    }
    fb[pixel_index] = c / float(n_samples);
}

int main() {
    // Image

    float aspect_ratio = 16.0 / 9.0;
    int image_width = 1200;
    int samples_per_pixel = 500;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    int num_pixels = image_width * image_height;

    // Create world

    int n_objects = 50;

    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

    curandState *d_world_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_world_rand_state, sizeof(curandState)));

    create_world<<<1, 1>>>(d_world, n_objects, d_world_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Create camera

    // Camera configuration parameters.
    float vfov = 40;
    point3 look_from(0, 1.5, 8);
    point3 look_at(0, 1, 0);
    vec3 vup(0, 1, 0);
    float defocus_angle = 1.0;
    float focus_dist = 6;

    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    init_camera<<<1, 1>>>(d_camera, image_width, image_height, vfov, look_from, look_at, vup,
                          defocus_angle, focus_dist);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render

    auto start = std::chrono::high_resolution_clock::now();

    // Allocate frame buffer
    size_t fb_size = 3 * num_pixels * sizeof(float);
    color *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    int tx = 16;
    int ty = 16;

    // Allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    // Render
    dim3 blocks((image_width + tx - 1) / tx, (image_height + ty - 1) / ty);
    dim3 threads(tx, ty);
    std::cout << "blocks.x = " << blocks.x << std::endl;
    std::cout << "blocks.y = " << blocks.y << std::endl;
    std::cout << "threads.x = " << threads.x << std::endl;
    std::cout << "threads.y = " << threads.y << std::endl;

    init_render<<<blocks, threads>>>(d_rand_state, image_width, image_height);
    render<<<blocks, threads>>>(fb, image_width, image_height, samples_per_pixel, d_camera, d_world,
                                d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1e6;

    // Write output image

    std::ofstream file;
    std::string file_name = "image.ppm";
    file.open(file_name, std::ios::trunc);

    if (file.is_open()) {
        // Output FB as .ppm image
        file << "P3\n" << image_width << " " << image_height << "\n255\n";
        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                size_t pixel_index = j * image_width + i;
                color pixel_color = fb[pixel_index];
                write_color(file, pixel_color);
            }
        }
        file.close();
        std::cout << "Successfully wrote to " << file_name << std::endl;
    } else {
        std::cerr << "Unable to open file: " << file_name << std::endl;
    }

    // Free memory
    free_world<<<1, 1>>>(d_world);
    checkCudaErrors(cudaGetLastError());
    free_camera<<<1, 1>>>(d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_world_rand_state));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    std::cout << "Rendered in " << duration.count() << " seconds" << std::endl;
}
