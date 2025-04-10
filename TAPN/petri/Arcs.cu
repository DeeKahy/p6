#include "Arcs.h"

/**
 * @brief Selects firing function based on input arc type
 * 
 * @param outputTokens a pointer to an array where the consumed tokens should go
 * @param outputCount the amount of tokens cosumed
 * @return tokens and how many tokens where consumed
 */
__device__ void Arc::fire(float *outputTokens, int *outputCount)
{
    switch (type)
    {
    case INPUT:
        printf("Input firing \n");
        inputFire(outputTokens, outputCount);
        break;
    case TRANSPORT:
        printf("Transport firing \n");
        transportFire(outputTokens, outputCount);
        break;
    case INHIBITOR:
        printf("Inhibitor firing \n");
        inhibitorFire(outputTokens, outputCount);
        break;
    }
}

/**
 * @brief 
 * 
 * @param result a pointer to a bool for indicating if an arc can fire
 * @return a boolean value indicating if an arc is able to fire
 */
__device__ void Arc::canFire(bool *result)
{
    bool tokensOk;
    bool invariantsOk;

    switch (type)
    {
    case INPUT:
        place->tokensHold(weight, timings, &tokensOk);
        place->invariantHold(weight, &invariantsOk);
        printf("tokesOk %d \n",tokensOk);
        printf("invariantsOk %d \n",invariantsOk);
        *result = tokensOk && invariantsOk;
        break;
    case TRANSPORT:
        place->tokensHold(weight, timings, &tokensOk);
        place->invariantHold(weight, &invariantsOk);
        *result = tokensOk && invariantsOk;
        break;
    case INHIBITOR:
        tokensOk = (place->tokenCount < constraint);
        place->invariantHold(weight, &invariantsOk);
        *result = tokensOk && invariantsOk;
        break;
    }
}

/**
 * @brief firing function for input transport arcs
 * 
 * @param outputTokens a pointer to an array where the consumed tokens should go
 * @param outputCount the amount of tokens cosumed
 * @return tokens and how many tokens where consumed
 */
__device__ void Arc::transportFire(float *outputTokens, int *outputCount)
{
    bool canFireResult = false;
    canFire(&canFireResult);

    if (canFireResult)
    {
        bool removeSuccess = false;
        place->removeTokens(weight, outputTokens);
        *outputCount = removeSuccess ? weight : 0;
    }
    else
    {
        *outputCount = 0;
    }
}

/**
 * @brief firing function for input arcs
 * 
 * @param outputTokens a pointer to an array where the consumed tokens should go
 * @param outputCount the amount of tokens cosumed
 * @return tokens and how many tokens where consumed
 */
__device__ void Arc::inputFire(float *outputTokens, int *outputCount)
{
    bool canFireResult = false;
    canFire(&canFireResult);

    printf("Input can fire \n");
    if (canFireResult)
    {
        for (size_t i = 0; i < weight; i++)
        {
            outputTokens[i] = 0.0f;
        }
        *outputCount = weight;

        float *dummy = new float[weight]{MAXFLOAT};
        bool removeSuccess = false;

        printf("Trying to remove\n");
        place->removeTokens(weight, dummy);
        if (!removeSuccess)
        {
            *outputCount = 0;
        }
    }
    else
    {
        *outputCount = 0;
    }
    printf("Input firing success\n");
}

/**
 * @brief firing function for inhibitor arcs
 *
 * this function needs more work to be fully up to spec
 * 
 * @param outputTokens a pointer to an array where the consumed tokens should go
 * @param outputCount the amount of tokens cosumed
 * @return tokens and how many tokens where consumed
 */
__device__ void Arc::inhibitorFire(float *outputTokens, int *outputCount)
{
    *outputCount = 0;
}

/**
 * @brief function for firing output arcs.
 * 
 * @param tokens an array of tokens to add to a place
 * @param tokenCount how many tokens are in the tokens array
 * @param success a pointer to a bool to indicate a success
 * @return bool
 */
__device__ void OutputArc::fire(float *tokens, int tokenCount, bool *success)
{
    if (isTransport)
    {
        if (tokenCount >= weight)
        {
            bool addSuccess = false;
            output->addTokens(tokens);
            *success = addSuccess;
        }
        else
        {
            *success = false;
        }
    }
    else
    {
        printf("Output firing \n");
        float *newTokens = new float[weight];
        for (size_t i = 0; i < weight; i++)
        {
            newTokens[i] = 0.0f;
        }
        bool addSuccess = false;
        output->addTokens(newTokens /* , weight, &addSuccess */);
        *success = addSuccess;
    }
}