#include "arc.h"

ArcPoint::ArcPoint(int id, double x, double y, bool isControlPoint) 
    : id(id), x(x), y(y), isControlPoint(isControlPoint) {}

Arc::Arc(const std::string& id, const std::string& sourceId, const std::string& targetId)
    : id(id), sourceId(sourceId), targetId(targetId), type("normal") {}

void Arc::addPoint(const ArcPoint& point) {
    arcPath.push_back(point);
}