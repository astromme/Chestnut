#ifndef CHESTNUT_DATAUTILS_H
#define CHESTNUT_DATAUTILS_H

#include <QPair>
#include <QStringList>

typedef QStringList Declarations;
typedef QStringList Executions;

typedef QPair<Declarations, Executions> ProgramStrings;

bool operator+(const ProgramStrings & p1, const ProgramStrings & p2) {
  return ProgramStrings(p1.first + p2.first, p1.second + p2.second);
}

#endif //CHESTNUT_DATAUTILS_H