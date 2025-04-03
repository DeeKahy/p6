#include "Place.h"

/**
 * Checks if the invariant in a place holds for tokenCount amount of tokens and returns
 * using pass by reference
 *
 * @param place reference to the place invariantHold works on
 * @param tokenCount the amount of tokens that needs to hold the invariant
 * @param returne a pointer to where the boolean result is returned
 *
 * @return Returns true if an invariant holds for tokenCount amount of tokens
 */
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

/**
 * Removes an amount of youngest tokens from a place
 *
 * @param place from which removeTokens should remove tokens from
 * @param amount of tokens to remove
 * @param removedTokens a pointer to the where the removed tokens should go
 *
 * @return returns the removed tokens into the removedTokens pointer
 */
__device__ void removeTokens(Place *place, int amount, float *removedTokens)
{
    for (size_t i = 0; i < amount; i++)
    {
        for (size_t j = 0; j < place->tokenCount; j++)
        {
            if (removedTokens[i] > place->tokens[j])
            {
                removedTokens[i] = place->tokens[j];
                place->tokens[j] = MAXFLOAT;
            }
        }
    }
}

/**
 * Checks if an amount of tokens hold the timing required by an arc
 *
 * @param place the place from which the tokens need to hold
 * @param amount of tokens that should hold
 * @param timing an array of two floates describing the timing eg. a - b
 * @param returne a pointer to a bool for returning the result
 *
 * @return if enough tokens hold the timing
 */
__device__ void tokensHold(Place *place, int amount, float timing[2], bool *returne)
{
    float minAge = timing[0];
    float maxAge = timing[1];

    int count{0};
    for (size_t i = 0; i < place->tokenCount; i++)
    {
        if (place->tokens[i] > minAge && place->tokens[i] < maxAge)
        {
            count++;
        }
    }
    *returne = count >= amount;
}

/**
 * Adds tokens to a place
 *
 * @param place to which the tokens should go
 * @param token to add to the place's tokens
 *
 * @return nothing
 */
__device__ void addTokens(Place *place, float *token)
{
    place->tokens[place->tokenCount++] = *token;
}