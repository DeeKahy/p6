#ifndef PETRI_NET_H
#define PETRI_NET_H

// Maximum string length constants
#define MAX_ID_LENGTH 256
#define MAX_NAME_LENGTH 512
#define MAX_FORMULA_LENGTH 2048
#define MAX_TYPE_LENGTH 64
#define MAX_INVARIANT_LENGTH 256
#define MAX_DISTRIBUTION_LENGTH 64
#define MAX_FIRING_MODE_LENGTH 64

// Maximum collection sizes
#define MAX_PLACES 256
#define MAX_TRANSITIONS 256
#define MAX_ARCS 512
#define MAX_QUERIES 32

// Forward declarations
class Arc;
class Place;
class Transition;

// Place class representing a place in the Petri net
class Place {
public:
    char id[MAX_ID_LENGTH];
    char name[MAX_NAME_LENGTH];
    int initialMarking;
    char invariant[MAX_INVARIANT_LENGTH];
    double positionX;
    double positionY;
    char type[MAX_TYPE_LENGTH];

    Place(const char* id);
    Place() : initialMarking(0), positionX(0), positionY(0) { 
        id[0] = '\0'; 
        name[0] = '\0'; 
        invariant[0] = '\0'; 
        type[0] = '\0'; 
    }
    ~Place() = default;
};

// Transition class representing a transition in the Petri net
class Transition {
public:
    char id[MAX_ID_LENGTH];
    char name[MAX_NAME_LENGTH];
    double positionX;
    double positionY;
    char distribution[MAX_DISTRIBUTION_LENGTH];
    double value;  // For constant distribution
    double a;      // For uniform distribution lower bound
    double b;      // For uniform distribution upper bound
    bool urgent;
    int priority;
    char firingMode[MAX_FIRING_MODE_LENGTH];

    Transition(const char* id);
    Transition() : positionX(0), positionY(0), value(0), a(0), b(0), 
                  urgent(false), priority(0) { 
        id[0] = '\0'; 
        name[0] = '\0'; 
        distribution[0] = '\0'; 
        firingMode[0] = '\0'; 
    }
    ~Transition() = default;
};

// Query class for storing query information
class Query {
public:
    char name[MAX_NAME_LENGTH];
    char type[MAX_TYPE_LENGTH];
    char formula[MAX_FORMULA_LENGTH];
    bool active;

    Query(const char* name);
    Query() : active(false) { 
        name[0] = '\0'; 
        type[0] = '\0'; 
        formula[0] = '\0'; 
    }
    ~Query() = default;
};

// PlaceMap structure replacing std::unordered_map for places
struct PlaceMap {
    Place places[MAX_PLACES];
    char keys[MAX_PLACES][MAX_ID_LENGTH];
    int size;

    PlaceMap() : size(0) {}

    Place* find(const char* id) {
        for (int i = 0; i < size; i++) {
            if (strcmp(keys[i], id) == 0) {
                return &places[i];
            }
        }
        return nullptr;
    }

    void insert(const Place& place) {
        if (size < MAX_PLACES) {
            strcpy(keys[size], place.id);
            places[size] = place;
            size++;
        }
    }
};

// TransitionMap structure replacing std::unordered_map for transitions
struct TransitionMap {
    Transition transitions[MAX_TRANSITIONS];
    char keys[MAX_TRANSITIONS][MAX_ID_LENGTH];
    int size;

    TransitionMap() : size(0) {}

    Transition* find(const char* id) {
        for (int i = 0; i < size; i++) {
            if (strcmp(keys[i], id) == 0) {
                return &transitions[i];
            }
        }
        return nullptr;
    }

    void insert(const Transition& transition) {
        if (size < MAX_TRANSITIONS) {
            strcpy(keys[size], transition.id);
            transitions[size] = transition;
            size++;
        }
    }
};

// PetriNet class representing the entire Petri net
class PetriNet {
public:
    char id[MAX_ID_LENGTH];
    char type[MAX_TYPE_LENGTH];
    char name[MAX_NAME_LENGTH];
    bool active;

    // Replace STL containers with arrays
    PlaceMap placeMap;
    TransitionMap transitionMap;
    Arc arcs[MAX_ARCS];
    int arcCount;
    Query queries[MAX_QUERIES];
    int queryCount;

    PetriNet(const char* id);
    PetriNet() : active(false), arcCount(0), queryCount(0) { 
        id[0] = '\0'; 
        type[0] = '\0'; 
        name[0] = '\0'; 
    }
    ~PetriNet() = default;

    void addPlace(const Place& place);
    void addTransition(const Transition& transition);
    void addArc(const Arc& arc);
    void addQuery(const Query& query);

    // Utility method to get place by ID
    Place* getPlace(const char* id);

    // Utility method to get transition by ID
    Transition* getTransition(const char* id);
};

#endif // PETRI_NET_H
