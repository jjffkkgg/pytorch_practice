/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

#include <casadi/mem.h>
int state_from_gz(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int state_from_gz_alloc_mem(void);
int state_from_gz_init_mem(int mem);
void state_from_gz_free_mem(int mem);
int state_from_gz_checkout(void);
void state_from_gz_release(int mem);
void state_from_gz_incref(void);
void state_from_gz_decref(void);
casadi_int state_from_gz_n_out(void);
casadi_int state_from_gz_n_in(void);
casadi_real state_from_gz_default_in(casadi_int i);
const char* state_from_gz_name_in(casadi_int i);
const char* state_from_gz_name_out(casadi_int i);
const casadi_int* state_from_gz_sparsity_in(casadi_int i);
const casadi_int* state_from_gz_sparsity_out(casadi_int i);
int state_from_gz_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
casadi_functions* state_from_gz_functions(void);
int quad_force_moment(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int quad_force_moment_alloc_mem(void);
int quad_force_moment_init_mem(int mem);
void quad_force_moment_free_mem(int mem);
int quad_force_moment_checkout(void);
void quad_force_moment_release(int mem);
void quad_force_moment_incref(void);
void quad_force_moment_decref(void);
casadi_int quad_force_moment_n_out(void);
casadi_int quad_force_moment_n_in(void);
casadi_real quad_force_moment_default_in(casadi_int i);
const char* quad_force_moment_name_in(casadi_int i);
const char* quad_force_moment_name_out(casadi_int i);
const casadi_int* quad_force_moment_sparsity_in(casadi_int i);
const casadi_int* quad_force_moment_sparsity_out(casadi_int i);
int quad_force_moment_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
casadi_functions* quad_force_moment_functions(void);
#ifdef __cplusplus
} /* extern "C" */
#endif