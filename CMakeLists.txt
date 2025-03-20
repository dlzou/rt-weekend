cmake_minimum_required(VERSION 3.20.0)

project(ray_tracer VERSION 0.1.0 LANGUAGES CUDA CXX)

add_executable(ray_tracer 
    camera.h
    disk.h
    color.h
    hittable.h
    hittable_list.h
    interval.h
    main.cu
    material.h
    ray.h
    rt_weekend.h
    sphere.h
    vec3.h
)
