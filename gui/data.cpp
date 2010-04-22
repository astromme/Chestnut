#include "data.h"

Data::Data(const QString& name)
  : Object(name, 0)
{
}

Data::~Data()
{

}

QString Data::tempData(Type t) {
  static int counter = 0;
  counter++;
  switch (t) {
    case Data::Value:
      return QString("tempScalar%1").arg(counter);
    case Data::DataBlock:
      return QString("tempVector%1").arg(counter);
  }
}
