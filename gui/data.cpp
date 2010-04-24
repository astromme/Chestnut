#include "data.h"


Data::Data(const QString& name, Data::Type type, const QString &dataType)
  : Object(0)
{
  m_type = type;
  m_datatype = dataType;
}

Data::~Data()
{

}

QString Data::name() const
{
  return m_name;
}

Data::Type Data::category() const
{
  return m_type;
}

QString Data::datatype() const
{
  return m_datatype;
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
