#include "object.h"

#include "sink.h"
#include "source.h"
#include <QDebug>
#include <QGraphicsSceneMouseEvent>

Object::Object(const QString &name, QGraphicsObject* parent)
  : QGraphicsObject(parent)
{
  m_name = name;
  m_moved = false;
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


void Object::mousePressEvent ( QGraphicsSceneMouseEvent* event )
{
  event->accept();
  m_moved = false;
}
void Object::mouseMoveEvent ( QGraphicsSceneMouseEvent* event )
{
  QPointF diff = event->pos() - event->lastPos();
  moveBy(diff.x(), diff.y());
  m_moved = true;
}
void Object::mouseReleaseEvent ( QGraphicsSceneMouseEvent* event )
{
  if (!m_moved) {
    setSelected(!isSelected());
  }
}

QVariant Object::itemChange ( QGraphicsItem::GraphicsItemChange change, const QVariant& value )
{
  if (change == ItemSelectedHasChanged) {
    update();
  }
  return QGraphicsItem::itemChange(change, value);
}
