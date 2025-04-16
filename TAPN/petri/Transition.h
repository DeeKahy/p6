#pragma once
#include "Arcs.h"
#include <curand_kernel.h>

enum DistributionTypes
{
    CONSTANT,
    UNIFORM,
};

struct Distribution
{
    DistributionTypes type;
    float a;
    float b;
    float c;
    __device__ void sample(float *result);
};

struct Transition
{
    Arc* inputArcs[5];
    int inputArcsCount{0};
    OutputArc* outputArcs[5];
    int outputArcsCount{0};
    Distribution* distribution;
    float firingTime{0.0f};
    bool urgent;
    int id;
    __device__ void fire(float *consumed, int consumedCount, int *consumedAmout);
    __device__ void isReady(bool *result);
};


