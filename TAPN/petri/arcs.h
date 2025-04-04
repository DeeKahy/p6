#pragma once
#include "Place.h"
#include "Arcs.cu"

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

    void fire(float *outputTokens, int *maxOutput);
    void canFire(bool *result);
    void transportFire(float *outputTokens, int *outputCount);
    void inputFire(float *outputTokens, int *outputCount);
    void inhibitorFire(float *outputTokens, int *outputCount);
};

struct OutputArc
{
    Place *output;
    size_t weight;
    bool isTransport;

    void fire(float *tokens, int tokenCount, bool *success);
}