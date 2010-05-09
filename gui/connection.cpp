#include "connection.h"

#include "source.h"
#include "sink.h"
#include "drawingutils.h"
#include "sizes.h"

#include <QPainter>
#include <QDebug>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>

using namespace Chestnut;


Connection::Connection(Source* source, Sink* sink)
  : QGraphicsObject(source),
  m_highlighted(false)
{
  // Only objects are selectable
  // Sources can be removed by dragging them off of their sink
  setFlag(ItemIsSelectable, false);
  
  m_source = source;
  source->addConnection(this);
  connect(source->parentObject(), SIGNAL(xChanged()), SLOT(updateEndpoints()));
  connect(source->parentObject(), SIGNAL(yChanged()), SLOT(updateEndpoints()));
  
  m_sink = sink;
  sink->setConnection(this);
  connect(sink->parentObject(), SIGNAL(xChanged()), SLOT(updateEndpoints()));
  connect(sink->parentObject(), SIGNAL(yChanged()), SLOT(updateEndpoints()));
}

Connection::Connection(Source* source) 
  : QGraphicsObject(source),
  m_highlighted(false)
{
  // Only objects are selectable
  // Sources can be removed by dragging them off of their sink
  setFlag(ItemIsSelectable, false);
  
  m_source = source;
  source->addConnection(this);
  connect(source->parentObject(), SIGNAL(xChanged()), SLOT(updateEndpoints()));
  connect(source->parentObject(), SIGNAL(yChanged()), SLOT(updateEndpoints()));
  
  // No sink means a partial connection.
  m_sink = 0;
  m_partialEndpoint = source->connectedCenter();
}

Connection::~Connection()
{
  if (m_source) {
    m_source->removeConnection(this);
  }
  if (m_sink) {
    m_sink->setConnection(0);
  }
}

int Connection::type() const
{
  return Type;
}

void Connection::setHighlighted(bool highlighted)
{
  m_highlighted = highlighted;
  update();
}

bool Connection::isPartial() const
{
  return (m_sink == 0);
}
void Connection::setSink(Sink* sink)
{
  prepareGeometryChange();
  m_highlighted = false;
  if (m_sink) {
    m_sink->setConnection(0);
    disconnect(m_sink->parentObject(), SIGNAL(xChanged()), this, SLOT(updateEndpoints()));
    disconnect(m_sink->parentObject(), SIGNAL(yChanged()), this, SLOT(updateEndpoints()));
  }
  m_sink = sink;
  if (sink) {
    sink->setConnection(this);
    connect(sink->parentObject(), SIGNAL(xChanged()), SLOT(updateEndpoints()));
    connect(sink->parentObject(), SIGNAL(yChanged()), SLOT(updateEndpoints()));
  }
}
void Connection::setEndpoint(const QPointF& scenePoint)
{
  prepareGeometryChange();
  m_partialEndpoint = mapFromScene(scenePoint);
}

QPointF Connection::endpoint() const
{
  return isPartial() ? m_partialEndpoint : mapFromItem(sink(), sink()->connectedCenter());
}

void Connection::updateEndpoints()
{
  prepareGeometryChange();
}

QRectF Connection::boundingRect() const
{
  // The rectangle must contain both the curve of the connection plus its endpoint shape.
  return path().boundingRect().united(endShape().boundingRect()).adjusted(-2, -2, 2, 2); // margin is 2
}

Source* Connection::source() const
{
  return m_source;
}
Sink* Connection::sink() const
{
  return m_sink;
}

QPainterPath Connection::path() const
{
  //qDebug() << "Path of Connection";
  
  // Create 2 Quadratic lines meeting in the center of the line that
  // intersects the start and the endpoint.
  QPointF start = mapFromItem(source(), source()->connectedCenter());
  QPointF end = endpoint();
  
  QLineF line = QLineF(start, end);
  QPointF midpoint = line.pointAt(0.5);
  
  QLineF firstLine = QLineF(start, midpoint);
  QLineF secondLine = QLineF(midpoint, end);
  
  QPointF controlPoint1 = firstLine.pointAt(0.5);
  QPointF controlPoint2 = secondLine.pointAt(0.5);
  
  // Semi-inaccurate way of forcing a certain curve depth. Looks decent
  controlPoint1 += QPointF(0, 10);
  controlPoint2 -= QPointF(0, 10);
  
  // Formalize the above into a painter path
  QPainterPath path;
  path.moveTo(start);
  path.quadTo(controlPoint1, midpoint);
  path.quadTo(controlPoint2, end);

  return path;
}

QPainterPath Connection::endShape() const
{
  // The endshape depends on the type of connection and needs to match
  // the source of the connection.
  QPainterPath p;
  switch (source()->format()) {
    case Data::DataBlock:
      p.moveTo(endpoint());
      p.addEllipse(endpoint(), Size::inputRadius, Size::inputRadius);
      return p;
      break;
    case Data::Value:
      return triangle(endpoint(), Size::inputHeight, Size::inputWidth);
      break;
  }
}

void Connection::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{  
  //qDebug() << "Painting Connection";
  QPen p;
  p.setWidth(2);
  painter->setPen(p);
  painter->drawPath(path());
  
  p.setWidth(1);
  if (m_highlighted) {
    p.setBrush(Qt::red);
  }
  painter->setPen(p);
  painter->drawPath(endShape());
}

void Connection::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
  // If we have clicked on the end shape:
  if (endShape().boundingRect().adjusted(-2, -2, 2, 2).contains(event->pos())) {
    // Disconnect from sink
    event->accept();
    setSink(0);
    setEndpoint(event->scenePos());
    return;
  }
  event->ignore();
}
void Connection::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
  // If the sink is disconnected, make it follow the mouse
  setEndpoint(mapToScene(event->pos()));
}
void Connection::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
  //TODO: Use a function to prevent code duplication here and in source.cpp
  // Check for sink under mouse pointer. If it exists, connect it
  foreach(QGraphicsItem *item, scene()->items(mapToScene(event->pos()))) {
    Sink* sink = qgraphicsitem_cast<Sink*>(item);
    if (sink && !sink->isConnected() && sink->allowedFormats().contains(source()->format())) {
      if (source()->parentObject()->isData()) {
        setSink(sink);
      } else if (source()->parentObject()->isFunction()) {
        if (sink->parentObject()->isData()) {
          setSink(sink);
        } else {
          // Create 'implicit' data TODO
          //if (format())
          //Data *temp = new 
        }
      }
      return;
    }
  }
  delete this;
}
