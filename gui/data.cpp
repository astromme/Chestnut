#include "data.h"
#include "sink.h"

#include <QRegExpValidator>

Data::Data(const QString& name, Data::Format format, const QString &datatype)
  : Object(0)
{
  m_format = format;
  m_datatype = datatype;
  m_name = name;
  
  // The name should only have valid C++ variable names. Enforce this
  // in the gui rather than in the compiler == nicer/better interface
  m_nameValidator = new QRegExpValidator(QRegExp("[a-zA-Z][a-zA-Z0-9_]*"), this);
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
void Data::setName(const QString& name)
{
  m_name = name;
  // name is displayed in gui, repaint
  update();
}


Data::Format Data::format() const
{
  return m_format;
}

QString Data::datatype() const
{
  return m_datatype;
}
void Data::setDatatype(const QString& datatype)
{
  m_datatype = datatype;
  // datatype is shown in gui, repaint
  update();
}

QString Data::tempData(Data::Format f) {
  // static variable ensures that for one program run
  // no two temp variable names are the same
  static int counter = 0;
  counter++;
  switch (f) {
    case Data::Value:
      return QString("tempScalar%1").arg(counter);
    case Data::DataBlock:
      return QString("tempVector%1").arg(counter);
  }
}
