#include "arc.h"
#include <cstring>

ArcPoint::ArcPoint(int id, double x, double y, bool isControlPoint)
    : id(id), x(x), y(y), isControlPoint(isControlPoint) {}

Arc::Arc(const char* arcId, const char* sourcId, const char* tgtId) : arcPathSize(0) {
    strncpy(id, arcId, MAX_ID_LENGTH - 1);
    id[MAX_ID_LENGTH - 1] = '\0';

    strncpy(sourceId, sourcId, MAX_ID_LENGTH - 1);
    sourceId[MAX_ID_LENGTH - 1] = '\0';

    strncpy(targetId, tgtId, MAX_ID_LENGTH - 1);
    targetId[MAX_ID_LENGTH - 1] = '\0';

    // Set default type
    strncpy(type, "normal", MAX_ID_LENGTH - 1);
    type[MAX_ID_LENGTH - 1] = '\0';

    // Initialize other strings
    inscription[0] = '\0';
    transportId[0] = '\0';
}

void Arc::addPoint(const ArcPoint& point) {
    if (arcPathSize < MAX_PATH_POINTS) {
        arcPath[arcPathSize] = point;
        arcPathSize++;
    }
}
