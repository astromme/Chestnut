#include "datautils.h"

#include <QFile>
#include <QTextStream>

// Some useful operators for ProgramStrings. They do exactly what you think they would with strings
const ProgramStrings operator+(const ProgramStrings& p1, const ProgramStrings& p2) {
  return ProgramStrings(p1.first + p2.first, p1.second + p2.second);
}

const ProgramStrings operator+=(const ProgramStrings& p1, const ProgramStrings& p2)
{
  return ProgramStrings(p1 + p2);
}

void writeToFile(QString fname, ProgramStrings prog)
{
  Declarations declarations = prog.first;
  Executions executions = prog.second;
  
  QFile file(fname);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
      return;

  QTextStream out(&file);
  
  foreach (QString dec, declarations){
    out << dec << "\n";
  }
  
  out << "\n";
  
  foreach (QString exec, executions){
    out << exec << "\n";
  }
}



 