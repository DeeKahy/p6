#include "Net.h"
struct Tapn
{
    Place *places;
    int placesCount{0};
    Transition *transitions;
    int transitionsCount{0};
    // observers
    uint steps{0};
    float currentTime{0.0f};
    // transitionFirings
    __device__ void addObserver();
    __device__ void notify_observers();
    __device__ void step(bool *result);
    __device__ void fireTransition();
    __device__ void firingCount();
    __device__ void run();
    __device__ void shouldContinue();
    __device__ void delay();
    __device__ void updateTokenAges(float *delay);
    __device__ void updateEnabledTransitions();
};
__device__ void Tapn::addObserver()
{
}
__device__ void Tapn::notify_observers()
{
}
__device__ void Tapn::step(bool *result)
{
    updateEnabledTransitions();

    

}
__device__ void Tapn::fireTransition()
{
}
__device__ void Tapn::firingCount()
{
}
__device__ void Tapn::run()
{

}
__device__ void Tapn::shouldContinue()
{
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
        for (size_t j = 0; j < places[i].tokenCount; j++)
        {
            places[i].tokens[j] += *delay;
        }
    }
}
__device__ void Tapn::updateEnabledTransitions()
{
    for (size_t i = 0; i < transitionsCount; i++)
    {
        bool ready;
        transitions[i].isReady(&ready);
    }
}