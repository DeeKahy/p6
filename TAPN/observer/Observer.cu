#include "Observer.h"

__device__ void TokenAgeObserver::onStep(const SimulationEvent *event)
{
    if (event->type == TOKENS_CHANGED)
    {
        auto &data = event->tokens;

        for (size_t i = 0; i < data.tokenCount; i++)
        {
            if (data.newTokens[i] > maxAllowedAge)
            {
                shouldStop = true;
                break;
            }
        }

        ageDistribution[data.placeId] = data.newTokens;
        ageCount[data.placeId] = data.tokenCount;
    }
}

__device__ void TokenAgeObserver::getShouldStop(bool *getShouldStop)
{
    *getShouldStop = getShouldStop;
} 

__device__ void TokenCountObserver::onStep(const SimulationEvent *event)
{
    if (event->type == TOKENS_CHANGED)
    {
        auto &data = event->tokens;

        for (size_t i = 0; i < numThresholds; i++)
        {
            auto &t = thresholds[i];
            if (t.placeId == data.placeId)
            {
                bool result = false;
                checkCondition(data.tokenCount, t.threshold, t.comparison, &result);
                
                if (result)
                {
                    shouldStop = true;
                    return;
                }
            }
        }
    }
}


__device__ void TokenCountObserver::checkCondition(unsigned count, unsigned threshold, Comparison comp, bool *result)
{
    switch (comp)
    {
    case LESS_THAN:
        *result = count < threshold;
        break;
    case LESS_EQUAL:
        *result = count <= threshold;
        break;
    case EQUAL:
        *result = count == threshold;
        break;
    case GREATER_EQUAL:
        *result = count >= threshold;
        break;
    case GREATER_THAN:
        *result = count > threshold;
        break;
    default: // Unknown comparison type
        *result = false;
        break;
    }
}

__device__ void TokenCountObserver::getShouldStop(bool *getShouldStop)
{
    // *getShouldStop = getShouldStop;
} 