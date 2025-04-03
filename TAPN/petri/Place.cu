#include "Place.h"
__device__ void invariantHold(Place *place, int tokenCount, bool *returne)
{
    if (tokenCount > place->tokenCount)
    {
        *returne = false;
        return;
    }
    if (&place->invariant == nullptr)
    {
        *returne = true;
        return;
    }
    int count{0};
    for (size_t i = 0; i < place->tokenCount; i++)
    {
        bool result;
        place->invariant.condition(&place->tokens[i], &result);
        if (result)
        {
            count++;
        }
    }

    *returne = count >= tokenCount;
}

__device__ void removeTokens(Place *place, int amount, float *removedTokens)
{
    for (size_t i = 0; i < amount; i++)
    {
        for (size_t j = 0; j < place->tokenCount; j++)
        {
            if (removedTokens[i] > place->tokens[j])
            {
                removedTokens[i] = place->tokens[j];
            }
        }
    }
}

__device__ void tokensHold(Place* place,int amount, float timing[2], bool *returne)
{
    float minAge = timing[0];
    float maxAge = timing[1];
    
    int count{0};
    for (size_t i = 0; i < place->tokenCount; i++)
    {
        if (place->tokens[i] > minAge && place->tokens[i] < maxAge) {
            count++;
        }
    }
    *returne = count >= amount;
}

__device__ void addTokens(Place *place, float *token)
{
    place->tokens[place->tokenCount++] = *token;
}