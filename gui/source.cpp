#include "source.h"

#include "sizes.h"
#include "drawingutils.h"
#include "object.h"
#include "sink.h"
#include "connection.h"

#include <QPainter>
#include <QDebug>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsScene>

using namespace Chestnut;

Source::Source(Data::Type type, Object* parent)
  : QGraphicsObject(parent)
{
  m_dataType = type;
  m_activeConnection = 0;
  m_parent = parent;
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

Object* Source::parentObject() const
{
  return m_parent;
}

QList<Sink*> Source::connectedSinks() const
{
  QList<Sink*> connectedSinks;
  foreach(Connection* c, m_connections) {
    if (!c->isPartial()) {
      connectedSinks.append(c->sink());
    }
  }
  return connectedSinks;
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
  QPointF center = QPointF(inputWidth/2, inputHeight/2);
  return center;
}

QRectF Source::boundingRect() const
{
  QPointF margin(1, 1);
  QPointF topLeft = QPointF(0, 0) - margin;
  QPointF bottomRight = QPointF(inputWidth, inputHeight) + margin;
  return QRectF(topLeft, bottomRight);
}
void Source::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  QPointF center(inputWidth/2, inputHeight/2);
  painter->setBrush(Qt::gray);
  switch (m_dataType) {
    case Data::Value:
      painter->drawPath(triangle(center, inputWidth, inputHeight));
      break;
   
    case Data::DataBlock:
      painter->drawEllipse(center, inputWidth/2, inputHeight/2);
      break;
      
    default:
      qDebug() << "Unhandled datatype" << m_dataType;
      break;
  }
}

void Source::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
  // allow us to get mouseMove/mouseRelease events
  event->accept();
  Connection *c = new Connection(this);
  m_activeConnection = c;
}

void Source::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
  m_activeConnection->setEndpoint(mapToScene(event->pos()));
}

void Source::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
  // Check for sink under mouse pointer. If it exists, connect it
  Sink *s = 0;
  foreach(QGraphicsItem *item, scene()->items(mapToScene(event->pos()))) {
    Sink* sink = qgraphicsitem_cast<Sink*>(item);
    if (sink && sink->allowedTypes().contains(dataType())) {
      m_activeConnection->setSink(sink);
      return;
    }
  }
  delete m_activeConnection;
  m_activeConnection = 0;
}






