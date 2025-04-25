// vec3.h
#ifndef VEC3_H
#define VEC3_H

#include <math.h>

static inline void vec3_set(double* x, double x0, double x1, double x2) {
    x[0] = x0; x[1] = x1; x[2] = x2;
}

static inline void vec3_copy(double* out, const double* in) {
    out[0] = in[0]; out[1] = in[1]; out[2] = in[2];
}

static inline void vec3_add(double* out, const double* a, const double* b) {
    for (int i = 0; i < 3; i++) out[i] = a[i] + b[i];
}

static inline void vec3_sub(double* out, const double* a, const double* b) {
    for (int i = 0; i < 3; i++) out[i] = a[i] - b[i];
}

static inline void vec3_scale(double* out, const double* v, double s) {
    for (int i = 0; i < 3; i++) out[i] = v[i] * s;
}

static inline void vec3_add_scaled(double* out, const double* a, const double* b, double s) {
    for (int i = 0; i < 3; i++) out[i] = a[i] + s * b[i];
}

static inline double vec3_dot(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline double vec3_norm(const double* v) {
    return sqrt(vec3_dot(v, v));
}

static inline void vec3_normalize(double* out, const double* in) {
    double norm = vec3_norm(in);
    if (norm > 0.0) {
        for (int i = 0; i < 3; i++) out[i] = in[i] / norm;
    } else {
        vec3_set(out, 0.0, 0.0, 0.0);
    }
}

#endif // VEC3_H

