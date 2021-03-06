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
  setFlag(ItemIsSelectable, false);
}

Source::~Source()
{
}

int Source::type() const
{
  return Type;
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
  foreach(Connection *connection, m_connections) {
    m_connections.removeAll(connection);
    delete connection;
  }
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
  
  // Check for sink under mouse pointer. If it exists, change color of connection endpoint
  Sink *s = 0;
  foreach(QGraphicsItem *item, scene()->items(mapToScene(QRectF(event->pos() - QPointF(5, 5), QSizeF(10, 10))))) {
    Sink* sink = qgraphicsitem_cast<Sink*>(item);
    if (sink && !sink->isConnected() && sink->allowedFormats().contains(format())) {
      if ((parentObject()->isData()) || (parentObject()->isFunction() && sink->parentObject()->isData())) {
        m_activeConnection->setHighlighted(true);
        return;
      }
    }
  }
  m_activeConnection->setHighlighted(false);
}

void Source::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
  // Check for sink under mouse pointer. If it exists, connect it
  Sink *s = 0;
  foreach(QGraphicsItem *item, scene()->items(mapToScene(QRectF(event->pos() - QPointF(5, 5), QSizeF(10, 10))))) {
    Sink* sink = qgraphicsitem_cast<Sink*>(item);
    if (sink && !sink->isConnected() && sink->allowedFormats().contains(format())) {
      if (parentObject()->isData()) {
        m_activeConnection->setSink(sink);
      } else if (parentObject()->isFunction()) {
        if (sink->parentObject()->isData()) {
          m_activeConnection->setSink(sink);
        } else {
          delete m_activeConnection;
          m_activeConnection = 0;
          // Create 'implicit' data TODO
          //if (format())
          //Data *temp = new 
        }
      }
      return;
    }
  }
  delete m_activeConnection;
  m_activeConnection = 0;
}
