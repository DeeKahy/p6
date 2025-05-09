#pragma once
#include "Invariant.h"
#include <string>
#include <cfloat>
struct Place
{
    int id;
    int tokenCount{0};
    float tokens[8] {FLT_MAX};
    
    Invariant *invariant{nullptr};
    int invariantCount{0};

    __device__ void tokensHold(int amount, float timing[2], bool *returne);
    __device__ void invariantHold(int tokenCount, bool *returne);
    __device__ void addTokens(float *token, int count);
    __device__ void removeTokens(int amount, float *removedTokens,int* count);
    __device__ void shiftTokens();
};