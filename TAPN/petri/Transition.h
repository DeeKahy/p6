#pragma once
#include "Arcs.h"
#include <curand_kernel.h>

enum DistributionTypes
{
    CONSTANT,
    UNIFORM,
    NORMAL,
    EXPONENTIAL,
};

struct Distribution
{
    DistributionTypes type;
    float a;
    float b;
    float c;

    // __device__ void init();
    __device__ void sample(curandState *state, float *result);
};

struct Transition
{

    Arc inputArcs[5];
    int inputArcsCount{0};
    OutputArc outputArcs[5];
    int outputArcsCount{0};
    Distribution distribution;
    float firingTime{FLT_MAX};
    bool urgent;
    int id;

    __device__ void fire(float *consumed, int consumedCount, int *consumedAmout, Place* realplaces);
    __device__ void isReady(bool *result, float *missing, Place* realplaces);
};
