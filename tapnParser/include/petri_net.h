#ifndef PETRI_NET_H
#define PETRI_NET_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// Forward declarations
class Arc;
class Place;
class Transition;

// Place class representing a place in the Petri net
class Place {
public:
    std::string id;
    std::string name;
    int initialMarking;
    std::string invariant;
    double positionX;
    double positionY;
    std::string type;

    Place(const std::string& id);
    ~Place() = default;
};

// Transition class representing a transition in the Petri net
class Transition {
public:
    std::string id;
    std::string name;
    double positionX;
    double positionY;
    std::string distribution;
    double value;  // For constant distribution
    double a;      // For uniform distribution lower bound
    double b;      // For uniform distribution upper bound
    bool urgent;
    int priority;
    std::string firingMode;

    Transition(const std::string& id);
    ~Transition() = default;
};

// Query class for storing query information
class Query {
public:
    std::string name;
    std::string type;
    std::string formula;
    bool active;

    Query(const std::string& name);
    ~Query() = default;
};

// PetriNet class representing the entire Petri net
class PetriNet {
public:
    std::string id;
    std::string type;
    std::string name;
    bool active;
    std::unordered_map<std::string, std::shared_ptr<Place>> places;
    std::unordered_map<std::string, std::shared_ptr<Transition>> transitions;
    std::vector<std::shared_ptr<Arc>> arcs;
    std::vector<std::shared_ptr<Query>> queries;

    PetriNet(const std::string& id);
    ~PetriNet() = default;

    void addPlace(std::shared_ptr<Place> place);
    void addTransition(std::shared_ptr<Transition> transition);
    void addArc(std::shared_ptr<Arc> arc);
    void addQuery(std::shared_ptr<Query> query);

    // Utility method to get place by ID
    std::shared_ptr<Place> getPlace(const std::string& id);

    // Utility method to get transition by ID
    std::shared_ptr<Transition> getTransition(const std::string& id);
};

#endif // PETRI_NET_H