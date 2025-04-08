#pragma once
#include "Place.h"

enum ArcType
{
    INPUT,
    TRANSPORT,
    INHIBITOR
};

struct Arc
{
    ArcType type;
    Place *place;
    size_t weight;
    float timings[2];
    size_t constraint;

    __device__ void fire(float *outputTokens, int *maxOutput);
    __device__ void canFire(bool *result);
    __device__ void transportFire(float *outputTokens, int *outputCount);
    __device__ void inputFire(float *outputTokens, int *outputCount);
    __device__ void inhibitorFire(float *outputTokens, int *outputCount);
};

struct OutputArc
{
    Place *output;
    size_t weight;
    bool isTransport;

    __device__ void fire(float *tokens, int tokenCount, bool *success);
};