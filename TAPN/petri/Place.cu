#include "Place.h"

/**
 * @brief Checks if the invariant in a place holds for tokenCount amount of tokens and returns using pass by reference
 *
 * @param tokenCount the amount of tokens that needs to hold the invariant
 * @param returne a pointer to where the boolean result is returned
 * @return Returns true if an invariant holds for tokenCount amount of tokens
 */
__device__ void Place::invariantHold(int tokenCount, bool *returne)
{
    if (tokenCount > this->tokenCount)
    {
        *returne = false;
        return;
    }

    if (invariant == nullptr)
    {
        *returne = true;
        return;
    }
    printf("not a nullptr\n");
    
    int count{0};
    for (size_t i = 0; i < this->tokenCount; i++)
    {
        bool result;
        this->invariant->condition(&this->tokens[i], &result);
        if (result)
        {
            count++;
        }
    }

    *returne = count >= tokenCount;
}

/**
 * @brief Removes an amount of youngest tokens from a place
 *
 * @param place from which removeTokens should remove tokens from
 * @param amount of tokens to remove
 * @param removedTokens a pointer to the where the removed tokens should go
 * @return returns the removed tokens into the removedTokens pointer
 */
__device__ void Place::removeTokens(int amount, float *removedTokens)
{
    printf("Removing tokens!!!\n");
    printf("amount %d", amount);
    for (size_t i = 0; i < amount; i++)
    {
        for (size_t j = 0; j < this->tokenCount; j++)
        {
            if (removedTokens[i] > this->tokens[j])
            {
                removedTokens[i] = this->tokens[j];
                this->tokens[j] = MAXFLOAT;
                this->tokenCount--;
                this->shiftTokens();
            }
        }
    }
}

/**
 * @brief Checks if an amount of tokens hold the timing required by an arc
 *
 * @param place the place from which the tokens need to hold
 * @param amount of tokens that should hold
 * @param timing an array of two floates describing the timing eg. a - b
 * @param returne a pointer to a bool for returning the result
 * @return if enough tokens hold the timing
 */
__device__ void Place::tokensHold(int amount, float timing[2], bool *returne)
{
    printf("Checking tokens\n");
    float minAge = timing[0];
    float maxAge = timing[1];

    int count{0};
    for (size_t i = 0; i < this->tokenCount; i++)
    {
        if (this->tokens[i] >= minAge && this->tokens[i] <= maxAge)
        {
            count++;
        }
    }
    printf("done checking tokens\n");
    *returne = count >= amount;
}

/**
 * @brief Adds tokens to a place
 *
 * @param place to which the tokens should go
 * @param token to add to the place's tokens
 *
 * @return nothing
 */
__device__ void Place::addTokens(float *token)
{
    printf("adding tokens \n");
    printf("token %f \n" ,*token);
    this->tokens[this->tokenCount++] = *token;
    printf("added tokens \n");
}

/**
 * @brief shifts tokens in a place's token array to the left
 *
 * @param place on which the tokens should be shifted
 * @return works on the array in place
 */
__device__ void Place::shiftTokens()
{
    bool needsToShift{false};
    int foundAt{0};
    // Check if the first tokens up to tokenCount are not tokens
    for (size_t i = 0; i < this->tokenCount; i++)
    {
        if (this->tokens[i] == MAXFLOAT)
        {
            foundAt = i;
            needsToShift = true;
        }
    }

    if (!needsToShift)
    {
        return;
    }

    for (size_t i = foundAt; i <= this->tokenCount; i++)
    {
        this->tokens[i] = this->tokens[i + 1];
    }
}