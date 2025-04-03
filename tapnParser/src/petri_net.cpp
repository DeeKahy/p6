#include "petri_net.h"
#include "arc.h"

Place::Place(const std::string& id) : id(id), initialMarking(0), positionX(0), positionY(0) {}

Transition::Transition(const std::string& id) : id(id), positionX(0), positionY(0), 
    value(0), a(0), b(0), urgent(false), priority(0) {}

Query::Query(const std::string& name) : name(name), active(false) {}

PetriNet::PetriNet(const std::string& id) : id(id), active(false) {}

void PetriNet::addPlace(std::shared_ptr<Place> place) {
    places[place->id] = place;
}

void PetriNet::addTransition(std::shared_ptr<Transition> transition) {
    transitions[transition->id] = transition;
}

void PetriNet::addArc(std::shared_ptr<Arc> arc) {
    arcs.push_back(arc);
}

void PetriNet::addQuery(std::shared_ptr<Query> query) {
    queries.push_back(query);
}

std::shared_ptr<Place> PetriNet::getPlace(const std::string& id) {
    auto it = places.find(id);
    if (it != places.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<Transition> PetriNet::getTransition(const std::string& id) {
    auto it = transitions.find(id);
    if (it != transitions.end()) {
        return it->second;
    }
    return nullptr;
}