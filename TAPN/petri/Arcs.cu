#include "Place.h"

enum InputArcType
{
    INPUT,
    TRANSPORT,
    INHIBITOR
};

struct arc
{
    InputArcType type;
    Place *place;
    size_t weight;
    float timings[2];
    size_t threshold;

    __device__ void fire()
    {
        switch (type)
        {
        case INPUT:
            break;
        case TRANSPORT:
            break;
        case INHIBITOR:
            break;
        }
    }
};
