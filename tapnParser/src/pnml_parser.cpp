
#include "pnml_parser.h"
#include "arc.h"
#include <iostream>
#include <string>
#include <sstream>

std::shared_ptr<PetriNet> PNMLParser::parse(const std::string& filename) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename.c_str());

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
    std::shared_ptr<PetriNet> petriNet = std::make_shared<PetriNet>(netNode.attribute("id").value());
    petriNet->type = netNode.attribute("type").value();
    petriNet->active = stringToBool(netNode.attribute("active").value(), false);

    // Parse places, transitions, and arcs
    parsePlaces(netNode, petriNet);
    parseTransitions(netNode, petriNet);
    parseArcs(netNode, petriNet);

    // Parse queries
    parseQueries(pnmlNode, petriNet);

    return petriNet;
}

void PNMLParser::parsePlaces(pugi::xml_node& netNode, std::shared_ptr<PetriNet> petriNet) {
    for (pugi::xml_node placeNode : netNode.children("place")) {
        std::string id = placeNode.attribute("id").value();
        auto place = std::make_shared<Place>(id);

        place->name = placeNode.attribute("name").value();
        place->initialMarking = stringToInt(placeNode.attribute("initialMarking").value(), 0);
        place->invariant = placeNode.attribute("invariant").value();
        place->positionX = stringToDouble(placeNode.attribute("positionX").value(), 0);
        place->positionY = stringToDouble(placeNode.attribute("positionY").value(), 0);

        // Get type information if available
        pugi::xml_node typeNode = placeNode.child("type");
        if (typeNode) {
            pugi::xml_node textNode = typeNode.child("text");
            if (textNode) {
                place->type = textNode.text().get();
            }
        }

        petriNet->addPlace(place);
    }
}

void PNMLParser::parseTransitions(pugi::xml_node& netNode, std::shared_ptr<PetriNet> petriNet) {
    for (pugi::xml_node transNode : netNode.children("transition")) {
        std::string id = transNode.attribute("id").value();
        auto transition = std::make_shared<Transition>(id);

        transition->name = transNode.attribute("name").value();
        transition->positionX = stringToDouble(transNode.attribute("positionX").value(), 0);
        transition->positionY = stringToDouble(transNode.attribute("positionY").value(), 0);
        transition->distribution = transNode.attribute("distribution").value();
        transition->value = stringToDouble(transNode.attribute("value").value(), 0);
        transition->a = stringToDouble(transNode.attribute("a").value(), 0);
        transition->b = stringToDouble(transNode.attribute("b").value(), 1);
        transition->urgent = stringToBool(transNode.attribute("urgent").value(), false);
        transition->priority = stringToInt(transNode.attribute("priority").value(), 0);
        transition->firingMode = transNode.attribute("firingMode").value();

        petriNet->addTransition(transition);
    }
}

void PNMLParser::parseArcs(pugi::xml_node& netNode, std::shared_ptr<PetriNet> petriNet) {
    for (pugi::xml_node arcNode : netNode.children("arc")) {
        std::string id = arcNode.attribute("id").value();
        std::string source = arcNode.attribute("source").value();
        std::string target = arcNode.attribute("target").value();

        auto arc = std::make_shared<Arc>(id, source, target);

        arc->type = arcNode.attribute("type").value();
        arc->inscription = arcNode.attribute("inscription").value();
        arc->transportId = arcNode.attribute("transportID").value();

        // Parse arc path points
        for (pugi::xml_node pathNode : arcNode.children("arcpath")) {
            int pathId = stringToInt(pathNode.attribute("id").value(), 0);
            double x = stringToDouble(pathNode.attribute("xCoord").value(), 0);
            double y = stringToDouble(pathNode.attribute("yCoord").value(), 0);
            bool isControlPoint = stringToBool(pathNode.attribute("arcPointType").value(), false);

            arc->addPoint(ArcPoint(pathId, x, y, isControlPoint));
        }

        petriNet->addArc(arc);
    }
}

void PNMLParser::parseQueries(pugi::xml_node& pnmlNode, std::shared_ptr<PetriNet> petriNet) {
    for (pugi::xml_node queryNode : pnmlNode.children("query")) {
        std::string name = queryNode.attribute("name").value();
        auto query = std::make_shared<Query>(name);

        query->type = queryNode.attribute("type").value();
        query->active = stringToBool(queryNode.attribute("active").value(), false);

        // Get formula if available
        pugi::xml_node formulaNode = queryNode.child("formula");
        if (formulaNode) {
            std::stringstream ss;
            formulaNode.print(ss);
            query->formula = ss.str();
        }

        petriNet->addQuery(query);
    }
}

double PNMLParser::stringToDouble(const std::string& str, double defaultValue) {
    if (str.empty()) return defaultValue;
    try {
        return std::stod(str);
    } catch (...) {
        return defaultValue;
    }
}

int PNMLParser::stringToInt(const std::string& str, int defaultValue) {
    if (str.empty()) return defaultValue;
    try {
        return std::stoi(str);
    } catch (...) {
        return defaultValue;
    }
}

bool PNMLParser::stringToBool(const std::string& str, bool defaultValue) {
    if (str.empty()) return defaultValue;
    return (str == "true" || str == "1");
}