#ifndef CAMERA_H
#define CAMERA_H

#include <curand_kernel.h>

class camera {
public:
    __device__ camera(int image_width, int image_height)
        : image_width(image_width), image_height(image_height) {
        float focal_length = 1.0f;
        float viewport_height = 2.0f;
        float viewport_width = viewport_height * (float(image_width) / image_height);
        camera_origin = point3(0, 0, 0);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        // We use right-handed coordinates, so x = right, y = up, z = into camera
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_du = viewport_u / image_width;
        pixel_dv = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left =
            camera_origin - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        pixel_origin = viewport_upper_left + 0.5 * (pixel_du + pixel_dv);
    }

    __device__ ray get_ray(int x, int y, curandState *rs) {
        point3 pixel = pixel_origin + (x + curand_uniform(rs) - 0.5) * pixel_du +
                       (y + curand_uniform(rs) - 0.5) * pixel_dv;
        return ray(camera_origin, pixel - camera_origin);
    }

    int image_width;
    int image_height;

private:
    point3 camera_origin;
    point3 pixel_origin;
    vec3 pixel_du;
    vec3 pixel_dv;
};

#endif
