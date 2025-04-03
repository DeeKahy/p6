#ifndef ARC_H
#define ARC_H

#include <string>
#include <memory>
#include <vector>
#include "petri_net.h"

// Point class for storing arc path points
class ArcPoint {
public:
    int id;
    double x;
    double y;
    bool isControlPoint;

    ArcPoint(int id, double x, double y, bool isControlPoint);
    ~ArcPoint() = default;
};

// Arc class representing an arc in the Petri net
class Arc {
public:
    std::string id;
    std::string sourceId;
    std::string targetId;
    std::string type;  // normal, inhibitor, transport
    std::string inscription;
    std::string transportId;
    std::vector<ArcPoint> arcPath;

    Arc(const std::string& id, const std::string& sourceId, const std::string& targetId);
    ~Arc() = default;

    void addPoint(const ArcPoint& point);
};

#endif // ARC_H