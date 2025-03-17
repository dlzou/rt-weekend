#ifndef CAMERA_H
#define CAMERA_H

#include <curand_kernel.h>

class camera {
public:
    __device__ camera(int image_width, int image_height, float vfov, point3 look_from,
                      point3 look_at, vec3 vup)
        : image_width(image_width), image_height(image_height), vfov(vfov) {
        camera_origin = look_from;

        // Determine viewport dimensions.
        float focal_length = (look_from - look_at).length();
        float theta = degrees_to_radians(vfov);
        float h = tan(theta / 2);
        float viewport_height = 2 * h * focal_length;
        float viewport_width = viewport_height * (float(image_width) / image_height);

        // Calculate the (u, v, w) unit basis vectors for the camera coordinate frame.
        // We use right-handed coordinates, so x = right, y = up, z = into camera.
        w = unit_vector(look_from - look_at); // z axis
        u = unit_vector(cross(vup, w));       // x axis
        v = cross(w, u);                      // y axis

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;   // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v; // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_du = viewport_u / image_width;
        pixel_dv = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        point3 viewport_upper_left =
            camera_origin - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
        pixel_origin = viewport_upper_left + 0.5 * (pixel_du + pixel_dv);
    }

    __device__ ray get_ray(int x, int y, curandState *rs) {
        point3 pixel = pixel_origin + (x + curand_uniform(rs) - 0.5) * pixel_du +
                       (y + curand_uniform(rs) - 0.5) * pixel_dv;
        return ray(camera_origin, pixel - camera_origin);
    }

    int image_width;
    int image_height;
    point3 camera_origin;
    float vfov = 90; // Vertical field of view
    point3 look_from = point3(0, 0, 0);
    point3 look_at = point3(0, 0, -1);
    vec3 vup = vec3(0, 1, 0);

private:
    point3 pixel_origin;
    vec3 pixel_du;
    vec3 pixel_dv;
    vec3 u, v, w;
};

#endif
