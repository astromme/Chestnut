#include "object.h"

#include "sink.h"
#include "source.h"
#include "connection.h"
#include <QDebug>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsScene>

Object::Object(QGraphicsObject* parent)
  : QGraphicsObject(parent)
{
  m_moved = false;
  m_visited = false;
  setFlag(ItemIsMovable);
  setFlag(ItemIsFocusable);
  setFlag(ItemIsSelectable);
}
Object::~Object()
{
  qDebug() << "Removing object" << sinks().length() << sources().length();
  foreach(Sink *sink, sinks()) {
    qDebug() << "Sink" << sink << "connected" << sink->isConnected();
    if (sink->isConnected()) {
      delete sink->connection();
    }
  }
  foreach(Source *source, sources()) {
    source->removeAllConnections();
  }
}

bool Object::isData() const
{
  return false;
}
bool Object::isFunction() const
{
  return false;
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
  qDebug() << "Mouse pressed on" << this;
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
    update();
  }
}

QVariant Object::itemChange ( QGraphicsItem::GraphicsItemChange change, const QVariant& value )
{
  if (change == ItemSelectedHasChanged) {
    if (isSelected()) {
      qDebug() << "Selected:" << this;
    } else {
      qDebug() << "Deselected:" << this;
    }
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
