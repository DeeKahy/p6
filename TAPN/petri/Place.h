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
};
void tokensHold(Place* place,int amount, float timing[2], bool *returne);
void invariantHold(Place *place, int tokenCount, bool *returne);
void addTokens(Place *place, float *token);
void removeTokens(Place *place, int amount, float* removedTokens);