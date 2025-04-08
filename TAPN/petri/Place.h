#pragma once
#include "Invariants.h"
#include <string>

struct Place
{
    int id;

    float tokens[8] {6.0f};
    int tokenCount;
    Invariant* invariant{nullptr};
    int invariantCount;

    __device__ void tokensHold(int amount, float timing[2], bool *returne);
    __device__ void invariantHold(int tokenCount, bool *returne);
    __device__ void addTokens(float *token);
    __device__ void removeTokens(int amount, float* removedTokens);
    __device__ void shiftTokens();
};