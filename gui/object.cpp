#include "object.h"

#include "sink.h"
#include "source.h"
#include <QDebug>
#include <QGraphicsSceneMouseEvent>

Object::Object(QGraphicsObject* parent)
  : QGraphicsObject(parent)
{
  m_moved = false;
  m_visited = false;
  setFlag(ItemIsMovable);
}
Object::~Object()
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

bool Object::isVisited() const
{
  return m_visited;
}
void Object::setVisited ( bool visited ) const
{
  Object* fakeThis = (Object*) this;
  fakeThis->m_visited = visited;
}
