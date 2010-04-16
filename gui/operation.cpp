#include "operation.h"

#include "object.h"

Operation::Operation(const QString& name, Object* parent)
  : QGraphicsObject(parent)
{
  m_name = name;
}
Operation::~Operation()
{

}

QString Operation::name() const
{
  return m_name;
}
