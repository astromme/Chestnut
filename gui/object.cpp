#include "object.h"

#include "sink.h"
#include <QDebug>

Object::Object(const QString &name, QGraphicsObject* parent)
  : QGraphicsObject(parent)
{
  m_name = name;
  setFlag(ItemIsMovable);
}
Object::~Object()
{

}

QString Object::name() const
{
  return m_name;
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
