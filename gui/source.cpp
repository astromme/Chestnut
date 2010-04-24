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

Source::Source(Data::Format format, Object* parent)
  : QGraphicsObject(parent)
{
  m_format = format;
  m_activeConnection = 0;
  m_parent = parent;
  connect(parent, SIGNAL(xChanged()), this, SLOT(moved()));
  connect(parent, SIGNAL(yChanged()), this, SLOT(moved()));
}
Data::Format Source::format() const
{
  return m_format;
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

QList< Data* > Source::connectedData() const
{
  QList< Data* > connected;
  foreach (Sink* sink, connectedSinks()){
    connected.append((Data*)sink->parentObject());
  }
  return connected;
}

QList< Function* > Source::connectedFunctions() const
{
  QList< Function* > connected;
  foreach (Sink* sink, connectedSinks()){
    connected.append((Function*)sink->parentObject());
  }
  return connected;
}


QPointF Source::connectedCenter() const
{
  QPointF center = QPointF(Size::inputWidth/2, Size::inputHeight/2);
  return center;
}

QRectF Source::rect() const {
  QPointF topLeft = QPointF(0, 0);
  QPointF bottomRight = QPointF(Size::inputWidth, Size::inputHeight);
  return QRectF(topLeft, bottomRight);
}

QRectF Source::boundingRect() const
{
  return rect().adjusted(-1, -1, 1, 1);
}
void Source::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  QPointF center(Size::inputWidth/2, Size::inputHeight/2);
  painter->setBrush(Qt::gray);
  switch ( m_format) {
    case Data::Value:
      painter->drawPath(triangle(center, Size::inputWidth, Size::inputHeight));
      break;
   
    case Data::DataBlock:
      painter->drawEllipse(center, Size::inputWidth/2, Size::inputHeight/2);
      break;
      
    default:
      qDebug() << "Unhandled datatype" << m_format;
      break;
  }
}

void Source::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
  if (parentObject()->isFunction() && m_connections.size() > 0) {
    event->ignore();
    return; // don't allow connections from functions to more than one data object
  }
  
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
    if (sink && !sink->isConnected() && sink->allowedFormats().contains(format())) {
      m_activeConnection->setSink(sink);
      return;
    }
  }
  delete m_activeConnection;
  m_activeConnection = 0;
}

void Source::moved() {
  //qDebug() << "source moved";
  if (m_activeConnection) {
    m_activeConnection->updateConnection();
  }
}
