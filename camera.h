#ifndef CAMERA_H
#define CAMERA_H

#include <curand_kernel.h>

class camera {
public:
    __device__ camera(int image_width, int image_height, float vfov, point3 look_from,
                      point3 look_at, vec3 vup, float defocus_angle, float focus_dist)
        : camera_origin(look_from), defocus_angle(defocus_angle) {
        // Determine viewport dimensions.
        float theta = degrees_to_radians(vfov);
        float h = tan(theta / 2);
        float viewport_height = 2 * h * focus_dist;
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
        pixel_u = viewport_u / image_width;
        pixel_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        point3 viewport_upper_left =
            camera_origin - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel_origin = viewport_upper_left + 0.5 * (pixel_u + pixel_v);

        // Calculate the camera defocus disk basis vectors.
        float defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ ray get_ray(int x, int y, curandState *rs) const {
        point3 pixel = pixel_origin + (x + curand_uniform(rs) - 0.5) * pixel_u +
                       (y + curand_uniform(rs) - 0.5) * pixel_v;
        point3 ray_origin = (defocus_angle <= 0) ? camera_origin : defocus_disk_sample(rs);
        return ray(ray_origin, pixel - ray_origin);
    }

private:
    point3 camera_origin;
    point3 pixel_origin;
    vec3 pixel_u;
    vec3 pixel_v;
    vec3 u, v, w; // Camera frame basis vectors

    float defocus_angle;
    vec3 defocus_disk_u; // Defocus disk horizontal radius
    vec3 defocus_disk_v; // Defocus disk vertical radius

    __device__ point3 defocus_disk_sample(curandState *rs) const {
        point3 p = random_in_unit_disk(rs);
        return camera_origin + p[0] * defocus_disk_u + p[1] * defocus_disk_v;
    }
};

#endif
