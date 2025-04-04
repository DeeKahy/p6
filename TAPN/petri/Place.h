#pragma once
#include "Invariants.h"
#include "Place.cu"
#include <string>

struct Place
{
    int id;
    string name;
    float tokens[8];
    int tokenCount;
    Invariant invariant{nullptr};
    int invariantCount;

    void tokensHold(int amount, float timing[2], bool *returne);
    void invariantHold(int tokenCount, bool *returne);
    void addTokens(float *token);
    void removeTokens(int amount, float* removedTokens);
    void shiftTokens();
};