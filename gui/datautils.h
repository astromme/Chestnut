#ifndef CHESTNUT_DATAUTILS_H
#define CHESTNUT_DATAUTILS_H

#include <QPair>
#include <QStringList>

typedef QStringList Declarations;
typedef QStringList Executions;

typedef QPair<Declarations, Executions> ProgramStrings;

const ProgramStrings operator+(const ProgramStrings & p1, const ProgramStrings & p2);
const ProgramStrings operator+=(const ProgramStrings & p1, const ProgramStrings & p2);

void writeToFile(QString fname, ProgramStrings prog);

#endif //CHESTNUT_DATAUTILS_H