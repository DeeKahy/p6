#ifndef PNML_PARSER_H
#define PNML_PARSER_H

#include "petri_net.h"
#include "pugixml.hpp"

class PNMLParser {
public:
    PNMLParser() = default;
    ~PNMLParser() = default;

    // Parse a PNML file and return a PetriNet object
    PetriNet* parse(const char* filename);

private:
    // Helper methods to parse specific elements
    void parsePlaces(pugi::xml_node& netNode, PetriNet* petriNet);
    void parseTransitions(pugi::xml_node& netNode, PetriNet* petriNet);
    void parseArcs(pugi::xml_node& netNode, PetriNet* petriNet);
    void parseQueries(pugi::xml_node& pnmlNode, PetriNet* petriNet);

    // Utility function to convert string to double with error handling
    double stringToDouble(const char* str, double defaultValue);

    // Utility function to convert string to int with error handling
    int stringToInt(const char* str, int defaultValue);

    // Utility function to convert string to bool with error handling
    bool stringToBool(const char* str, bool defaultValue);
};

#endif // PNML_PARSER_H
