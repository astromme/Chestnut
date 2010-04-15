#include "source.h"

#include "drawingutils.h"
#include "object.h"
#include "sink.h"
#include "connection.h"

#include <QPainter>
#include <QDebug>

Source::Source(Data::Type type, Object* parent)
  : QGraphicsItem(parent)
{
  m_dataType = type;
  m_width = 8;
  m_height = 8;
}
Data::Type Source::dataType() const
{
  return m_dataType;
}

Connection* Source::connectToSink(Sink* sink)
{
  Connection *c = new Connection(this, sink);
  return c;
}

void Source::addConnection(Connection* connection)
{
  m_connections.append(connection);
}
void Source::removeConnection(Connection* connection)
{
  m_connections.removeAll(connection);
}
void Source::removeAllConnections()
{
  m_connections.clear();
}

QPointF Source::connectedCenter() const
{
  QPointF center = QPointF(m_width/2, m_height/2);
  return center;
}

QRectF Source::boundingRect() const
{
  QPointF margin(1, 1);
  QPointF topLeft = QPointF(0, 0) - margin;
  QPointF bottomRight = QPointF(m_width, m_height) + margin;
  return QRectF(topLeft, bottomRight);
}
void Source::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  QPointF center(m_width/2, m_height/2);
  painter->setBrush(Qt::gray);
  switch (m_dataType) {
    case Data::Value:
      drawTriangle(painter, center, m_width, m_height);
      break;
   
    case Data::DataBlock:
      painter->drawEllipse(center, m_width/2, m_height/2);
      break;
      
    default:
      qDebug() << "Unhandled datatype" << m_dataType;
      break;
  }
}




