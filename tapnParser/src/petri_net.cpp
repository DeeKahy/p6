#include "petri_net.h"
#include "arc.h"
#include <cstring>

Place::Place(const char* placeId) {
    strncpy(id, placeId, MAX_ID_LENGTH - 1);
    id[MAX_ID_LENGTH - 1] = '\0';

    name[0] = '\0';
    invariant[0] = '\0';
    type[0] = '\0';
    initialMarking = 0;
    positionX = 0;
    positionY = 0;
}

Transition::Transition(const char* transId) {
    strncpy(id, transId, MAX_ID_LENGTH - 1);
    id[MAX_ID_LENGTH - 1] = '\0';

    name[0] = '\0';
    distribution[0] = '\0';
    firingMode[0] = '\0';
    positionX = 0;
    positionY = 0;
    value = 0;
    a = 0;
    b = 0;
    urgent = false;
    priority = 0;
}

Query::Query(const char* queryName) {
    strncpy(name, queryName, MAX_NAME_LENGTH - 1);
    name[MAX_NAME_LENGTH - 1] = '\0';

    type[0] = '\0';
    formula[0] = '\0';
    active = false;
}

PetriNet::PetriNet(const char* netId) : arcCount(0), queryCount(0), active(false) {
    strncpy(id, netId, MAX_ID_LENGTH - 1);
    id[MAX_ID_LENGTH - 1] = '\0';

    type[0] = '\0';
    name[0] = '\0';
}

void PetriNet::addPlace(const Place& place) {
    placeMap.insert(place);
}

void PetriNet::addTransition(const Transition& transition) {
    transitionMap.insert(transition);
}

void PetriNet::addArc(const Arc& arc) {
    if (arcCount < MAX_ARCS) {
        arcs[arcCount] = arc;
        arcCount++;
    }
}

void PetriNet::addQuery(const Query& query) {
    if (queryCount < MAX_QUERIES) {
        queries[queryCount] = query;
        queryCount++;
    }
}

Place* PetriNet::getPlace(const char* id) {
    return placeMap.find(id);
}

Transition* PetriNet::getTransition(const char* id) {
    return transitionMap.find(id);
}
