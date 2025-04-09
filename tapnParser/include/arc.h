#ifndef ARC_H
#define ARC_H

#include "petri_net.h"

// Maximum string length constants
#define MAX_ID_LENGTH 256
#define MAX_PATH_POINTS 128

// Point class for storing arc path points
class ArcPoint {
public:
    int id;
    double x;
    double y;
    bool isControlPoint;

    ArcPoint(int id, double x, double y, bool isControlPoint);
    ArcPoint() : id(0), x(0), y(0), isControlPoint(false) {}
    ~ArcPoint() = default;
};

// Arc class representing an arc in the Petri net
class Arc {
public:
    char id[MAX_ID_LENGTH];
    char sourceId[MAX_ID_LENGTH];
    char targetId[MAX_ID_LENGTH];
    char type[MAX_ID_LENGTH];  // normal, inhibitor, transport
    char inscription[MAX_ID_LENGTH];
    char transportId[MAX_ID_LENGTH];
    ArcPoint arcPath[MAX_PATH_POINTS];
    int arcPathSize;

    Arc(const char* id, const char* sourceId, const char* targetId);
    Arc() : arcPathSize(0) { 
        id[0] = '\0'; 
        sourceId[0] = '\0'; 
        targetId[0] = '\0'; 
        type[0] = '\0'; 
        inscription[0] = '\0'; 
        transportId[0] = '\0'; 
    }
    ~Arc() = default;

    void addPoint(const ArcPoint& point);
};

#endif // ARC_H
