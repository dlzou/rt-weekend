#include "utils.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "interval.h"
#include "material.h"
#include "sphere.h"

#include <curand_kernel.h>
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

__global__ void render_init(curandState *rand_state, int iw, int ih) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= iw) || (j >= ih))
        return;

    int pixel_index = j * iw + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void create_world(hittable **obj_list, hittable **world, camera **cam, int iw, int ih,
                             int n_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        material *mat_ground = new lambertian(color(0.8, 0.8, 0.0));
        material *mat1 = new metal(color(0.8, 0.8, 0.8), 0.3);
        material *mat2 = new lambertian(color(0.1, 0.2, 0.5));
        material *mat3 = new metal(color(0.8, 0.6, 0.2), 1);

        // These dereferenced assignments to dynamic objects require double pointers.
        *(obj_list) = new sphere(point3(0, -100.5, -1), 100, mat_ground);
        *(obj_list + 1) = new sphere(point3(-1, 0, -1), 0.5, mat1);
        *(obj_list + 2) = new sphere(point3(0, 0, -1.2), 0.5, mat2);
        *(obj_list + 3) = new sphere(point3(1, 0, -1), 0.5, mat3);
        *world = new hittable_list(obj_list, n_objects);

        *cam = new camera(iw, ih);
    }
}

__device__ color ray_color(const ray &r, const hittable **world, curandState *rs) {
    ray cur_ray = r;
    color cur_attenuation = color(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, interval(0.001, INFINITY), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered, rs)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return color(0, 0, 0);
            }
        } else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float a = 0.5f * (unit_direction.y() + 1.0f);
            color bg = (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
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

    float aspect_ratio = 16.0f / 9.0f;
    int image_width = 400;
    int samples_per_pixel = 100;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    int num_pixels = image_width * image_height;

    // Create world

    int n_objects = 4;

    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, n_objects * sizeof(hittable *)));

    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    create_world<<<1, 1>>>(d_list, d_world, d_camera, image_width, image_height, n_objects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render

    // Allocate frame buffer
    size_t fb_size = 3 * num_pixels * sizeof(float);
    color *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    int tx = 16;
    int ty = 16;

    // Allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    // Render our buffer
    dim3 blocks((image_width + tx - 1) / tx, (image_height + ty - 1) / ty);
    dim3 threads(tx, ty);
    std::clog << "blocks.x = " << blocks.x << "\n";
    std::clog << "blocks.y = " << blocks.y << "\n";
    std::clog << "threads.x = " << threads.x << "\n";
    std::clog << "threads.y = " << threads.y << "\n";

    render_init<<<blocks, threads>>>(d_rand_state, image_width, image_height);
    render<<<blocks, threads>>>(fb, image_width, image_height, samples_per_pixel, d_camera, d_world,
                                d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output FB as .ppm image
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            color pixel_color = fb[pixel_index];
            write_color(std::cout, pixel_color);
        }
    }

    // Free memory
    checkCudaErrors(cudaFree(fb));
    // TODO: free CUDA dynamic memory

    std::clog << "Done.\n";
}
