#include "connection.h"

#include "source.h"
#include "sink.h"

#include <QPainter>
#include <QDebug>

Connection::Connection(Source* source, Sink* sink)
  : QGraphicsItem(source)
{
  m_source = source;
  m_sink = sink;
  source->addConnection(this);
  sink->setConnection(this);
}

QRectF Connection::boundingRect() const
{
  QPointF start = mapFromItem(source(), source()->connectedCenter());
  QPointF end = mapFromItem(sink(), sink()->connectedCenter());

  return QRectF(start, end);
}

Source* Connection::source() const
{
  return m_source;
}
Sink* Connection::sink() const
{
  return m_sink;
}

void Connection::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  QPointF start = mapFromItem(source(), source()->connectedCenter());
  QPointF end = mapFromItem(sink(), sink()->connectedCenter());
  
  QLineF line = QLineF(start, end);
  QPointF midpoint = line.pointAt(0.5);
  
  QLineF firstLine = QLineF(start, midpoint);
  QLineF secondLine = QLineF(midpoint, end);
  
  QPointF controlPoint1 = firstLine.pointAt(0.5);
  QPointF controlPoint2 = secondLine.pointAt(0.5);
  
  controlPoint1 += QPointF(0, 10);
  controlPoint2 -= QPointF(0, 10);
  
  QPainterPath path;
  path.moveTo(start);
  path.quadTo(controlPoint1, midpoint);
  path.quadTo(controlPoint2, end);
  painter->drawPath(path);
}
