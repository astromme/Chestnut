#include "object.h"

Object::Object(QGraphicsObject* parent)
  : QGraphicsObject(parent)
{
  
}

QList< Source* > Object::sources() const
{
  return m_sources;
}
QList< Sink* > Object::sinks() const
{
  return m_sinks;
}


void Object::setSources(QList< Source* > sources)
{

}
void Object::setSinks(QList< Sink* > sinks)
{

}
