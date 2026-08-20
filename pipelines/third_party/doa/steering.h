#ifndef STEERING_H
#define STEERING_H

#include "srp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* geometry API */
ArrayGeometry* array_geometry_create(int M);
ArrayGeometry* array_geometry_create_uca(int M, float radius);
ArrayGeometry* array_geometry_create_ula(int M, float spacing);
ArrayGeometry* array_geometry_create_custom(int M, const float* x, const float* y);
void array_geometry_destroy(ArrayGeometry* g);

/* angle / steering API */
float* srp_create_uniform_angles(int num_angles);

Complex*** srp_build_steering(
    const SRP_Config* cfg,
    const ArrayGeometry* geom,
    const float* angles
);
void srp_destroy_steering(
    Complex*** steering,
    int num_angles,
    int microphones
);

#ifdef __cplusplus
}
#endif

#endif
