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
    // //printf("%d",type);
    switch (type)
    {
    case INPUT:
        // //printf("Input firing \n");
        inputFire(outputTokens, outputCount);
        break;
    case TRANSPORT:
        // printf("Transport firing \n");
        transportFire(outputTokens, outputCount);
        break;
    case INHIBITOR:
        // //printf("Inhibitor firing \n");
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
__device__ void Arc::canFire(bool *result, float *missing)
{
    bool tokensOk;
    bool invariantsOk;
    switch (type)
    {
    case INPUT:
        place->tokensHold(weight, timings, &tokensOk, missing);
        place->invariantHold(weight, &invariantsOk);
        // printf("tokesOk input%d \n",tokensOk);
        // printf("invariantsOk input %d \n",invariantsOk);
        *result = tokensOk && invariantsOk;
        break;
    case TRANSPORT:
        place->tokensHold(weight, timings, &tokensOk, missing);
        place->invariantHold(weight, &invariantsOk);
        // //printf("tokesOk transport %d \n",tokensOk);
        // //printf("invariantsOk  transpot %d \n",invariantsOk);
        *result = tokensOk && invariantsOk;
        break;
    case INHIBITOR:
        tokensOk = (place->tokenCount < constraint);
        // place->invariantHold(weight, &invariantsOk);
        *result = tokensOk; //&& invariantsOk;
        break;
    default:
        // //printf("test");
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
    // bool canFireResult = false;
    // float missing{0};
    // canFire(&canFireResult, &missing);
    // if (canFireResult)
    // {
        int count{0};
        place->removeTokens(weight, outputTokens, &count);
        // printf("remvoed this many tokens %d", count);
        *outputCount = count > 0 ? count : 0;
    // }
    // else
    // {
    //     *outputCount = 0;
    // }
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
    // float missing{0};
    // bool canFireResult = false;
    // canFire(&canFireResult, &missing);

    // // //printf("Input can fire \n");
    // if (canFireResult)
    // {
        for (size_t i = 0; i < weight; i++)
        {
            outputTokens[i] = FLT_MAX;
        }
        *outputCount = weight;

        float *dummy = new float[weight]{FLT_MAX};
        bool removeSuccess = false;

        // //printf("Trying to remove\n");
        int count{0};
        place->removeTokens(weight, dummy, &count);
        if (!removeSuccess)
        {
            *outputCount = 0;
        }
        *outputCount = count;
        delete[] dummy;
    // }
    // else
    // {
    //     *outputCount = 0;
    // }
    // //printf("Input firing success\n");
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
    // //printf("\n token count%d\n",tokenCount);
    if (isTransport)
    {
        if (tokenCount >= weight)
        {
            bool addSuccess = false;
            // for (size_t i = 0; i < tokenCount; i++)
            // {
            //     output->addTokens(&tokens[i]);
            // }
            // for (size_t i = 0; i < tokenCount; i++)
            // {

            //     tokens[i]+= 1.0f;
            // }
            output->addTokens(tokens, tokenCount);

            *success = true;
        }
        else
        {
            *success = false;
        }
    }
    else
    {
        // //printf("Output firing \n");
        float newToken = 0.0f;
        bool addSuccess = false;
        output->addTokens(&newToken /* , weight, &addSuccess */, 1);
        *success = addSuccess;
    }
}