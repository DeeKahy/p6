
#ifndef PNML_PARSER_H
#define PNML_PARSER_H

#include <string>
#include <memory>
#include "petri_net.h"
#include "pugixml.hpp"

class PNMLParser {
public:
    PNMLParser() = default;
    ~PNMLParser() = default;

    // Parse a PNML file and return a PetriNet object
    std::shared_ptr<PetriNet> parse(const std::string& filename);

private:
    // Helper methods to parse specific elements
    void parsePlaces(pugi::xml_node& netNode, std::shared_ptr<PetriNet> petriNet);
    void parseTransitions(pugi::xml_node& netNode, std::shared_ptr<PetriNet> petriNet);
    void parseArcs(pugi::xml_node& netNode, std::shared_ptr<PetriNet> petriNet);
    void parseQueries(pugi::xml_node& pnmlNode, std::shared_ptr<PetriNet> petriNet);

    // Utility function to convert string to double with error handling
    double stringToDouble(const std::string& str, double defaultValue);

    // Utility function to convert string to int with error handling
    int stringToInt(const std::string& str, int defaultValue);

    // Utility function to convert string to bool with error handling
    bool stringToBool(const std::string& str, bool defaultValue);
};

#endif // PNML_PARSER_H