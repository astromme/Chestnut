#include "data.h"
#include "sink.h"


Data::Data(const QString& name, Data::Format format, const QString &datatype)
  : Object(0)
{
  m_format = format;
  m_datatype = datatype;
  m_name = name;
}

Data::~Data()
{

}

int Data::type() const
{
  return Type;
}


bool Data::isData() const
{
  return true;
}

bool Data::isInitialized() const
{
  // if sinks to data have no connections, 
  // then nothing is feeding into them
  if (sinks().size() == 0) {
    return true; // no sinks so it is required to be init
  }
  if (sinks()[0]->isConnected()) {
    return false;
  }
  return true;
}

void Data::setExpression(QString expr)
{
  m_expression = expr;
}


QString Data::expression() const
{
  return m_expression;
}


QString Data::name() const
{
  return m_name;
}

Data::Format Data::format() const
{
  return m_format;
}

QString Data::datatype() const
{
  return m_datatype;
}

QString Data::tempData(Data::Format f) {
  static int counter = 0;
  counter++;
  switch (f) {
    case Data::Value:
      return QString("tempScalar%1").arg(counter);
    case Data::DataBlock:
      return QString("tempVector%1").arg(counter);
  }
}
