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
    // Place *place;
    int place;
    size_t weight{1};
    float timings[2];
    size_t constraint;

    __device__ void fire(float *outputTokens, int *maxOutput, Place *places);
    __device__ void canFire(bool *result, float *missing, Place *places);
    __device__ void transportFire(float *outputTokens, int *outputCount, Place *places);
    __device__ void inputFire(float *outputTokens, int *outputCount, Place *places);
    __device__ void inhibitorFire(float *outputTokens, int *outputCount);
};

struct OutputArc
{
    // Place *output;
    int output;
    size_t weight{1};
    bool isTransport{false};

    __device__ void fire(float *tokens, int tokenCount, bool *success, Place *places);
};