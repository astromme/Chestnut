#include "sink.h"

#include "drawingutils.h"
#include "connection.h"
#include "source.h"
#include "object.h"

#include <QPainter>
#include <QDebug>


Sink::Sink(Data::Types allowedTypes, Object* parent)
  : QGraphicsObject(parent)
{
  m_allowedTypes = allowedTypes;
  m_connection = 0;
  m_internalMargin = 2;
  m_width = 8;
  m_height = 8;
}
Data::Types Sink::allowedTypes() const
{
  return m_allowedTypes;
}

int Sink::type() const
{
  return Type;
}

void Sink::setConnection(Connection* connection)
{
  if (!connection) {
    m_connection = connection;
    return;
  }
  
  Data::Type type = connection->source()->dataType();
  Q_ASSERT(allowedTypes().contains(type));
  
  m_connection = connection;
  m_connectionType = type;
}

Connection* Sink::connection() const
{
  return m_connection;
}

QPointF Sink::connectedCenter()
{
  if (!m_connection) {
    return QPointF();
  }
  
  int location = m_allowedTypes.indexOf(m_connectionType);
  QPointF center(m_width/2, m_height/2);
  center += QPointF(location*(m_internalMargin + m_width), 0);
  return center;
}

QRectF Sink::boundingRect() const
{
  QPointF margin(1, 1);
  qreal totalWidth = 0;
  foreach(Data::Type type, m_allowedTypes) {
    totalWidth += m_width;
  }
  totalWidth += m_internalMargin*m_allowedTypes.length();
  
  QPointF topLeft = QPointF(0, 0) - margin;
  QPointF bottomRight = QPointF(totalWidth, m_height) + margin;
  return QRectF(topLeft, bottomRight);
}
void Sink::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  QPen p(Qt::black, 1, Qt::DotLine);
  painter->setPen(p);
  
  QPointF topLeft = QPointF(0, 0);
  
  foreach(Data::Type type, m_allowedTypes) {
    QPointF center = topLeft + QPointF(m_width/2, m_height/2);
    switch (type) {
      case Data::Value:
        painter->drawPath(triangle(center, m_width, m_height));
        break;
    
      case Data::DataBlock:
        painter->drawEllipse(center, m_width/2, m_height/2);
        break;
        
      default:
        qDebug() << "Unhandled datatype" << type;
        break;
    }
  topLeft = QPointF(topLeft.x() + m_width + m_internalMargin, topLeft.y());
  }
}
