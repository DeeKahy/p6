
#include <iostream>
#include <memory>
#include <string>
#include "pnml_parser.h"

#include "arc.h"
void printNetInfo(const std::shared_ptr<PetriNet>& net) {
    std::cout << "Petri Net Information:" << std::endl;
    std::cout << "ID: " << net->id << std::endl;
    std::cout << "Type: " << net->type << std::endl;
    std::cout << "Active: " << (net->active ? "true" : "false") << std::endl;
    std::cout << std::endl;

    std::cout << "Places (" << net->places.size() << "):" << std::endl;
    for (const auto& [id, place] : net->places) {
        std::cout << "  Place ID: " << place->id << std::endl;
        std::cout << "    Name: " << place->name << std::endl;
        std::cout << "    Initial Marking: " << place->initialMarking << std::endl;
        std::cout << "    Position: (" << place->positionX << ", " << place->positionY << ")" << std::endl;
        if (!place->type.empty()) {
            std::cout << "    Type: " << place->type << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Transitions (" << net->transitions.size() << "):" << std::endl;
    for (const auto& [id, transition] : net->transitions) {
        std::cout << "  Transition ID: " << transition->id << std::endl;
        std::cout << "    Name: " << transition->name << std::endl;
        std::cout << "    Position: (" << transition->positionX << ", " << transition->positionY << ")" << std::endl;
        std::cout << "    Distribution: " << transition->distribution << std::endl;
        if (transition->distribution == "constant") {
            std::cout << "    Value: " << transition->value << std::endl;
        } else if (transition->distribution == "uniform") {
            std::cout << "    Range: [" << transition->a << ", " << transition->b << "]" << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Arcs (" << net->arcs.size() << "):" << std::endl;
    for (const auto& arc : net->arcs) {
        std::cout << "  Arc ID: " << arc->id << std::endl;
        std::cout << "    Source: " << arc->sourceId << std::endl;
        std::cout << "    Target: " << arc->targetId << std::endl;
        std::cout << "    Type: " << arc->type << std::endl;
        std::cout << "    Inscription: " << arc->inscription << std::endl;
        if (!arc->transportId.empty()) {
            std::cout << "    Transport ID: " << arc->transportId << std::endl;
        }
        std::cout << "    Path Points: " << arc->arcPath.size() << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Queries (" << net->queries.size() << "):" << std::endl;
    for (const auto& query : net->queries) {
        std::cout << "  Query Name: " << query->name << std::endl;
        std::cout << "    Type: " << query->type << std::endl;
        std::cout << "    Active: " << (query->active ? "true" : "false") << std::endl;
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pnml_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    PNMLParser parser;

    try {
        std::shared_ptr<PetriNet> petriNet = parser.parse(filename);
        if (petriNet) {
            printNetInfo(petriNet);
            std::cout << "Successfully parsed PNML file: " << filename << std::endl;
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