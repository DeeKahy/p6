#include "Net.h"
__device__ void Tapn::addObserver(SimulationObserver *observer)
{

    cudaMalloc((void **)&observers, sizeof(*observers) + sizeof(*observer));
    observers[observersCount] = observer;
    observersCount++;
}

__device__ void Tapn::notify_observers(const SimulationEvent *event)
{
    for (size_t i = 0; i < observersCount; i++)
    {
        observers[i]->onStep(event);
    }
}

__device__ void Tapn::step(bool *result)
{
    updateEnabledTransitions();

    EnabledTransition *enabled = new EnabledTransition[transitionsCount];
    int enabledCount = 0;

    for (int i = 0; i < transitionsCount; i++)
    {
        bool isReady = true;
        transitions[i]->isReady(&isReady);
        if (isReady)
        {
            enabled[enabledCount] = {i, transitions[i]->firingTime};
            enabledCount++;
        }
    }
    if (enabledCount == 0)
    {
        float ageIncrease{1.0f};
        updateTokenAges(&ageIncrease);
        failedAttempt++;
        steps++;
        delete[] enabled;
        if (failedAttempt >= 10)
        {
            *result = false;
        }
        else
        {
            *result = true;
        }
        return;
    }

    int urgentTransitionIndex = -1;
    for (size_t i = 0; i < enabledCount; i++)
    {
        int transitionIndex = enabled[i].index;
        if (transitions[transitionIndex]->urgent)
        {
            urgentTransitionIndex = transitionIndex;
            break;
        }
    }

    if (urgentTransitionIndex != -1)
    {
        fireTransition(urgentTransitionIndex, result);
        delete[] enabled;
        return;
    }

    EnabledTransition lowestFiringTime = {-1, FLT_MAX};
    for (size_t i = 0; i < enabledCount; i++)
    {
        int transitionIndex = enabled[i].index;
        if (transitions[transitionIndex]->firingTime < lowestFiringTime.firingTime)
        {
            lowestFiringTime = enabled[i];
        }
    }

    bool success = false;
    fireTransition(lowestFiringTime.index, &success);
    delete[] enabled;
}

__device__ void Tapn::fireTransition(size_t index, bool *result)
{
    float firingTime = transitions[index]->firingTime;

    SimulationEvent event;
    event.type = TRANSITION_FIRING;
    event.firing = {(int)index, firingTime};
    SimulationEvent preEvent = event;
    notify_observers(&preEvent);
    // currentTime += firingTime;
    float consumed[8]{FLT_MAX};
    int consumedCount{8};
    int consumedAmount;
    transitions[index]->fire(consumed, consumedCount, &consumedAmount);

    transitionFirings[index]++;
    steps++;


    *result = true;
}

/**
 * @brief Query function for seeing how many times a transition has fired
 *
 * @param index the transition to query
 * @param result a pointer to where the result should go
 * @return result
 */
__device__ void Tapn::firingCount(int index, int *result)
{
    *result = transitionFirings[index];
}

__device__ void Tapn::run()
{
    bool result;
    shouldContinue(&result);
    while (result)
    {
        step(&result);
        if (steps >= 30)
        {
            return;
        }
    }
}
__device__ void Tapn::run2(bool *success)
{
    bool result{true};
    while (result)
    {
        // for (size_t i = 0; i < placesCount; i++)
        // {
        //     // 0 &waiting0,
        //     // 1 &waiting1,
        //     // 2 &waiting2,
        //     // 3 &waiting3,
        //     // 4 &charging0,
        //     // 5 &charging1,
        //     // 6 &charging2,
        //     // 7 &charging3,
        //     // 8 &charged0,
        //     // 9 &charged1,
        //     // 10&charged2,
        //     // 11&charged3,
        //     // 12&chargedSum,
        //     // 13&flashing};
        //     switch (i)
        //     {
        //     case 0:
        //         printf("waiting 0:");
        //         break;
        //     case 1:
        //         printf("waiting 1:");
        //         break;
        //     case 2:
        //         printf("waiting 2:");
        //         break;
        //     case 3:
        //         printf("waiting 3:");
        //         break;
        //     case 4:
        //         printf("charging 0:");
        //         break;
        //     case 5:
        //         printf("charging 1:");
        //         break;
        //     case 6:
        //         printf("charging 2:");
        //         break;
        //     case 7:
        //         printf("charging 3:");
        //         break;
        //     case 8:
        //         printf("charged 0:");
        //         break;
        //     case 9:
        //         printf("charged 1:");
        //         break;
        //     case 10:
        //         printf("charged 2:");
        //         break;
        //     case 11:
        //         printf("charged 3:");
        //         break;
        //     case 12:
        //         printf("charged Sum:");
        //         break;
        //     case 13:
        //         printf("flashing:");
        //         break;
        //     default:
        //         printf("unknown state:");
        //         break;
        //     }
        //     for (size_t j = 0; j < places[i]->tokenCount; j++)
        //     {

        //         printf(" token number :%d value: %f", j, places[i]->tokens[j]);
        //     }
        //     printf("\n");
        // }
        // if ((places[4]->tokenCount + places[5]->tokenCount +
        //          places[6]->tokenCount + places[7]->tokenCount >=
        //      1) &&
        //     places[13]->tokenCount == 1 &&
        //     places[0]->tokenCount == 0 && places[1]->tokenCount == 0 &&
        //     places[2]->tokenCount == 0 && places[3]->tokenCount == 0)

        if ((places[4]->tokenCount + places[5]->tokenCount +
                 places[6]->tokenCount + places[7]->tokenCount ==
             1) &&
            places[13]->tokenCount == 1 &&
            places[0]->tokenCount == 0 && places[1]->tokenCount == 0 &&
            places[2]->tokenCount == 0 && places[3]->tokenCount == 0)

        // if (places[4]->tokenCount == 1 && places[5]->tokenCount == 1 && places[6]->tokenCount == 1 && places[7]->tokenCount == 1 &&
        //     places[13]->tokenCount == 1 &&
        //     places[0]->tokenCount == 0 && places[1]->tokenCount == 0 && places[2]->tokenCount == 0 && places[3]->tokenCount == 0)

        {
            *success = true;
            // printf("_______________________________________________SUCCESS__________________________");
            return;
        }

        if (currentTime <= 30)
        {
            step(&result);
        }
        else
        {

            *success = false;
            return;
        }
    }
}

__device__ void Tapn::shouldContinue(bool *result)
{
    for (size_t i = 0; i < observersCount; i++)
    {
        observers[i]->getShouldStop(result);
        if (*result)
        {
            return;
        }
    }
    *result = true;
}

__device__ void Tapn::delay()
{
}

__device__ void Tapn::updateTokenAges(float *delay)
{

    if (*delay <= 0.0f)
    {
        return;
    }
    currentTime += *delay;
    for (size_t i = 0; i < placesCount; i++)
    {
        for (size_t j = 0; j < places[i]->tokenCount; j++)
        {
            places[i]->tokens[j] += *delay;
        }
    }
}

__device__ void Tapn::updateEnabledTransitions()
{
    bool ready{true};
    for (size_t i = 0; i < transitionsCount; i++)
    {
        transitions[i]->isReady(&ready);
    }
}