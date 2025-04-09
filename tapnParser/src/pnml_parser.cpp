#include "pnml_parser.h"
#include "arc.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

PetriNet* PNMLParser::parse(const char* filename) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename);

    if (!result) {
        std::cerr << "Error parsing PNML file: " << result.description() << std::endl;
        return nullptr;
    }

    // Get the root node <pnml>
    pugi::xml_node pnmlNode = doc.child("pnml");
    if (!pnmlNode) {
        std::cerr << "Error: No <pnml> root node found" << std::endl;
        return nullptr;
    }

    // Get the <net> node
    pugi::xml_node netNode = pnmlNode.child("net");
    if (!netNode) {
        std::cerr << "Error: No <net> node found" << std::endl;
        return nullptr;
    }

    // Create a new PetriNet object
    PetriNet* petriNet = new PetriNet(netNode.attribute("id").value());

    // Set net attributes
    strncpy(petriNet->type, netNode.attribute("type").value(), MAX_TYPE_LENGTH - 1);
    petriNet->type[MAX_TYPE_LENGTH - 1] = '\0';

    petriNet->active = stringToBool(netNode.attribute("active").value(), false);

    // Parse places, transitions, and arcs
    parsePlaces(netNode, petriNet);
    parseTransitions(netNode, petriNet);
    parseArcs(netNode, petriNet);

    // Parse queries
    parseQueries(pnmlNode, petriNet);

    return petriNet;
}

void PNMLParser::parsePlaces(pugi::xml_node& netNode, PetriNet* petriNet) {
    for (pugi::xml_node placeNode : netNode.children("place")) {
        const char* id = placeNode.attribute("id").value();
        Place place(id);

        strncpy(place.name, placeNode.attribute("name").value(), MAX_NAME_LENGTH - 1);
        place.name[MAX_NAME_LENGTH - 1] = '\0';

        place.initialMarking = stringToInt(placeNode.attribute("initialMarking").value(), 0);

        strncpy(place.invariant, placeNode.attribute("invariant").value(), MAX_INVARIANT_LENGTH - 1);
        place.invariant[MAX_INVARIANT_LENGTH - 1] = '\0';

        place.positionX = stringToDouble(placeNode.attribute("positionX").value(), 0);
        place.positionY = stringToDouble(placeNode.attribute("positionY").value(), 0);

        // Get type information if available
        pugi::xml_node typeNode = placeNode.child("type");
        if (typeNode) {
            pugi::xml_node textNode = typeNode.child("text");
            if (textNode) {
                strncpy(place.type, textNode.text().get(), MAX_TYPE_LENGTH - 1);
                place.type[MAX_TYPE_LENGTH - 1] = '\0';
            }
        }

        petriNet->addPlace(place);
    }
}

void PNMLParser::parseTransitions(pugi::xml_node& netNode, PetriNet* petriNet) {
    for (pugi::xml_node transNode : netNode.children("transition")) {
        const char* id = transNode.attribute("id").value();
        Transition transition(id);

        strncpy(transition.name, transNode.attribute("name").value(), MAX_NAME_LENGTH - 1);
        transition.name[MAX_NAME_LENGTH - 1] = '\0';

        transition.positionX = stringToDouble(transNode.attribute("positionX").value(), 0);
        transition.positionY = stringToDouble(transNode.attribute("positionY").value(), 0);

        strncpy(transition.distribution, transNode.attribute("distribution").value(), MAX_DISTRIBUTION_LENGTH - 1);
        transition.distribution[MAX_DISTRIBUTION_LENGTH - 1] = '\0';

        transition.value = stringToDouble(transNode.attribute("value").value(), 0);
        transition.a = stringToDouble(transNode.attribute("a").value(), 0);
        transition.b = stringToDouble(transNode.attribute("b").value(), 1);
        transition.urgent = stringToBool(transNode.attribute("urgent").value(), false);
        transition.priority = stringToInt(transNode.attribute("priority").value(), 0);

        strncpy(transition.firingMode, transNode.attribute("firingMode").value(), MAX_FIRING_MODE_LENGTH - 1);
        transition.firingMode[MAX_FIRING_MODE_LENGTH - 1] = '\0';

        petriNet->addTransition(transition);
    }
}

void PNMLParser::parseArcs(pugi::xml_node& netNode, PetriNet* petriNet) {
    for (pugi::xml_node arcNode : netNode.children("arc")) {
        const char* id = arcNode.attribute("id").value();
        const char* source = arcNode.attribute("source").value();
        const char* target = arcNode.attribute("target").value();

        Arc arc(id, source, target);

        strncpy(arc.type, arcNode.attribute("type").value(), MAX_ID_LENGTH - 1);
        arc.type[MAX_ID_LENGTH - 1] = '\0';

        strncpy(arc.inscription, arcNode.attribute("inscription").value(), MAX_ID_LENGTH - 1);
        arc.inscription[MAX_ID_LENGTH - 1] = '\0';

        strncpy(arc.transportId, arcNode.attribute("transportID").value(), MAX_ID_LENGTH - 1);
        arc.transportId[MAX_ID_LENGTH - 1] = '\0';

        // Parse arc path points
        for (pugi::xml_node pathNode : arcNode.children("arcpath")) {
            int pathId = stringToInt(pathNode.attribute("id").value(), 0);
            double x = stringToDouble(pathNode.attribute("xCoord").value(), 0);
            double y = stringToDouble(pathNode.attribute("yCoord").value(), 0);
            bool isControlPoint = stringToBool(pathNode.attribute("arcPointType").value(), false);

            arc.addPoint(ArcPoint(pathId, x, y, isControlPoint));
        }

        petriNet->addArc(arc);
    }
}

void PNMLParser::parseQueries(pugi::xml_node& pnmlNode, PetriNet* petriNet) {
    for (pugi::xml_node queryNode : pnmlNode.children("query")) {
        const char* name = queryNode.attribute("name").value();
        Query query(name);

        strncpy(query.type, queryNode.attribute("type").value(), MAX_TYPE_LENGTH - 1);
        query.type[MAX_TYPE_LENGTH - 1] = '\0';

        query.active = stringToBool(queryNode.attribute("active").value(), false);

        // Get formula if available
        pugi::xml_node formulaNode = queryNode.child("formula");
        if (formulaNode) {
            std::stringstream ss;
            formulaNode.print(ss);
            const std::string formulaStr = ss.str();
            strncpy(query.formula, formulaStr.c_str(), MAX_FORMULA_LENGTH - 1);
            query.formula[MAX_FORMULA_LENGTH - 1] = '\0';
        }

        petriNet->addQuery(query);
    }
}

double PNMLParser::stringToDouble(const char* str, double defaultValue) {
    if (!str || str[0] == '\0') return defaultValue;
    try {
        return atof(str);
    } catch (...) {
        return defaultValue;
    }
}

int PNMLParser::stringToInt(const char* str, int defaultValue) {
    if (!str || str[0] == '\0') return defaultValue;
    try {
        return atoi(str);
    } catch (...) {
        return defaultValue;
    }
}

bool PNMLParser::stringToBool(const char* str, bool defaultValue) {
    if (!str || str[0] == '\0') return defaultValue;
    return (strcmp(str, "true") == 0 || strcmp(str, "1") == 0);
}
