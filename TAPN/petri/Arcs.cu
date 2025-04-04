#include "arcs.h"

__device__ void Arc::fire(float *outputTokens, int *outputCount)
{
    switch (type)
    {
    case INPUT:
        inputFire(outputTokens, outputCount);
        break;
    case TRANSPORT:
        transportFire(outputTokens, outputCount);
        break;
    case INHIBITOR:
        inhibitorFire(outputTokens, outputCount);
        break;
    }
}

__device__ void Arc::canFire(bool *result)
{
    bool tokensOk;
    bool invariantsOk;

    switch (type)
    {
    case INPUT:
        break;
    case TRANSPORT:
        this->place->tokensHold(this->weight, this->timings, &tokensOk);
        this->place->invariantHold(this->weight, &invariantsOk);
        *result = tokensOk && invariantsOk;
        break;
    case INHIBITOR:
        tokensOk = (place->tokenCount < constraint);
        this->place->invariantHold(this->weight, &invariantsOk);
        *result = tokensOk && invariantsOk;
        break;
    }
}

__device__ void Arc::transportFire(float *outputTokens, int *outputCount)
{
    bool canFireResult = false;
    canFire(&canFireResult);

    if (canFireResult)
    {
        bool removeSuccess = false;
        place->removeTokens(weight, outputTokens, &removeSuccess);
        *outputCount = removeSuccess ? weight : 0;
    }
    else
    {
        *outputCount = 0;
    }
}

__device__ void Arc::inputFire(float *outputTokens, int *outputCount)
{
    bool canFireResult = false;
    canFire(&canFireResult);

    if (canFireResult)
    {
        for (size_t i = 0; i < weight; i++)
        {
            outputTokens[i] = 0.0f;
        }
        *outputCount = weight;

        float dummy[weight];
        bool removeSuccess = false;
        place->removeTokens(weight, dummy, &removeSuccess);
        if (!removeSuccess)
        {
            *outputCount = 0;
        }
    }
    else
    {
        *outputCount = 0;
    }
}

__device__ void Arc::inhibitorFire(float *outputTokens, int *outputCount)
{
    *outputCount = 0;
}

__device__ void OutputArc::fire(float *tokens, int tokenCount, bool *success)
{
    if(isTransport) {
        if (tokenCount >= weight) 
        {
            bool addSuccess = false;
            output->addTokens(tokens, weight, &addSuccess);
            *success = addSuccess;
        } else {
            *success = false;
        }
    } else {
        float newTokens[weight];
        for (size_t i = 0; i < weight; i++)
        {
            newTokens[i] = 0.0f;
        }
        bool addSuccess = false;
        output->addTokens(newTokens, weight, &addSuccess);
        *success = addSuccess;
    }
}