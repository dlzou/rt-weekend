cmake_minimum_required(VERSION 3.20.0)

project(ray_tracer VERSION 0.1.0 LANGUAGES CUDA CXX)

add_executable(ray_tracer 
    camera.h
    color.h
    utils.h
    hittable.h
    hittable_list.h
    interval.h
    main.cu
    material.h
    ray.h
    sphere.h
    vec3.h
)
