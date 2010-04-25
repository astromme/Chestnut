#include "ParseLibs.h"

double totalTime(timeval* start, timeval* stop)
{
    return (stop->tv_sec + stop->tv_usec*0.000001)
      - (start->tv_sec + start->tv_usec*0.000001);
}
