enum SimulationEventType
{
    TRANSITION_FIRING,
    TRANSITION_FIRED,
    TOKENS_CHANGED,
    TIME_ADVANCED
};

struct TransitionFiringData
{
    int transition_id;
    float firing_time;
};

struct TokensChangedData
{
    int placeId;
    float *newTokens;
    int tokenCount;
};

struct SimulationEvent
{
    SimulationEventType type;
    union
    {
        TransitionFiringData firing;
        TokensChangedData tokens;
    };
};

struct SimulationObserver
{
    enum ObserverType
    {
        AGE_OBSERVER,
        COUNT_OBSERVER
    } type;
    bool shouldStop{false};

    __device__ virtual void onStep(const SimulationEvent *event) = 0;
    __device__ virtual void onCompletion() = 0;
    __device__ virtual void getShouldStop(bool *getShouldStop) = 0;
};

struct TokenAgeObserver : SimulationObserver
{
    float maxAllowedAge;
    float *ageDistribution[10]; // Needs a constant value telling the maximum amount of places allowed or something
    int ageCount[10];           // Same here

    __device__ TokenAgeObserver(float maxAge)
    {
        type = AGE_OBSERVER;
        maxAllowedAge = maxAge;
        shouldStop = false;
    }

    __device__ void onStep(const SimulationEvent *event);
    __device__ void onCompletion();
    __device__ void getShouldStop(bool *getShouldStop);
};

enum Comparison
{
    LESS_THAN,
    LESS_EQUAL,
    EQUAL,
    GREATER_EQUAL,
    GREATER_THAN
};

struct PlaceThreshold
{
    int placeId;
    unsigned threshold;
    Comparison comparison;
};

struct TokenCountObserver : SimulationObserver
{
    PlaceThreshold thresholds[10]; // Needs a constant value telling the maximum amount of places allowed or something
    int numThresholds;

    __device__ TokenCountObserver() {
        type = COUNT_OBSERVER;
        shouldStop = false;
        numThresholds = 0;
    }

    __device__ void add_threshold(int place_id, unsigned threshold, Comparison comp);
    __device__ void checkCondition(unsigned count, unsigned threshold, Comparison comp, bool *result);
    __device__ void onStep(const SimulationEvent *event);
    __device__ void getShouldStop(bool *getShouldStop);
};