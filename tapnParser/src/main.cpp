#include <iostream>
#include <cstring>
#include "pnml_parser.h"

void printNetInfo(const PetriNet* net) {
    std::cout << "Petri Net Information:" << std::endl;
    std::cout << "ID: " << net->id << std::endl;
    std::cout << "Type: " << net->type << std::endl;
    std::cout << "Active: " << (net->active ? "true" : "false") << std::endl;
    std::cout << std::endl;

    std::cout << "Places (" << net->placeMap.size << "):" << std::endl;
    for (int i = 0; i < net->placeMap.size; i++) {
        const Place& place = net->placeMap.places[i];
        std::cout << "  Place ID: " << place.id << std::endl;
        std::cout << "    Name: " << place.name << std::endl;
        std::cout << "    Initial Marking: " << place.initialMarking << std::endl;
        std::cout << "    Position: (" << place.positionX << ", " << place.positionY << ")" << std::endl;
        if (place.type[0] != '\0') {
            std::cout << "    Type: " << place.type << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Transitions (" << net->transitionMap.size << "):" << std::endl;
    for (int i = 0; i < net->transitionMap.size; i++) {
        const Transition& transition = net->transitionMap.transitions[i];
        std::cout << "  Transition ID: " << transition.id << std::endl;
        std::cout << "    Name: " << transition.name << std::endl;
        std::cout << "    Position: (" << transition.positionX << ", " << transition.positionY << ")" << std::endl;
        std::cout << "    Distribution: " << transition.distribution << std::endl;
        if (strcmp(transition.distribution, "constant") == 0) {
            std::cout << "    Value: " << transition.value << std::endl;
        } else if (strcmp(transition.distribution, "uniform") == 0) {
            std::cout << "    Range: [" << transition.a << ", " << transition.b << "]" << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Arcs (" << net->arcCount << "):" << std::endl;
    for (int i = 0; i < net->arcCount; i++) {
        const Arc& arc = net->arcs[i];
        std::cout << "  Arc ID: " << arc.id << std::endl;
        std::cout << "    Source: " << arc.sourceId << std::endl;
        std::cout << "    Target: " << arc.targetId << std::endl;
        std::cout << "    Type: " << arc.type << std::endl;
        std::cout << "    Inscription: " << arc.inscription << std::endl;
        if (arc.transportId[0] != '\0') {
            std::cout << "    Transport ID: " << arc.transportId << std::endl;
        }
        std::cout << "    Path Points: " << arc.arcPathSize << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Queries (" << net->queryCount << "):" << std::endl;
    for (int i = 0; i < net->queryCount; i++) {
        const Query& query = net->queries[i];
        std::cout << "  Query Name: " << query.name << std::endl;
        std::cout << "    Type: " << query.type << std::endl;
        std::cout << "    Active: " << (query.active ? "true" : "false") << std::endl;
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pnml_file>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];
    PNMLParser parser;

    try {
        PetriNet* petriNet = parser.parse(filename);
        if (petriNet) {
            printNetInfo(petriNet);
            std::cout << "Successfully parsed PNML file: " << filename << std::endl;
            delete petriNet;  // Clean up memory
        } else {
            std::cerr << "Failed to parse PNML file." << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
