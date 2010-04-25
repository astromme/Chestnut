#include "sink.h"

#include "sizes.h"
#include "drawingutils.h"
#include "connection.h"
#include "source.h"
#include "object.h"

#include <QPainter>
#include <QDebug>

using namespace Chestnut;

Sink::Sink(Data::Formats allowedFormats, Object* parent)
  : QGraphicsObject(parent)
{
  m_allowedFormats = allowedFormats;
  m_connection = 0;
  m_internalMargin = 2;
  m_parent = parent;
  setFlag(ItemIsSelectable, false);
  connect(parent, SIGNAL(xChanged()), this, SLOT(moved()));
  connect(parent, SIGNAL(yChanged()), this, SLOT(moved()));
}
Sink::Sink(Data::Format allowedFormat, Object* parent)
  : QGraphicsObject(parent)
{
  Data::Formats allowedFormats;
  allowedFormats << allowedFormat;
  
  // Important! Code copied from above constructor Sink::Sink()
  m_allowedFormats = allowedFormats;
  m_connection = 0;
  m_internalMargin = 2;
  m_parent = parent;
  setFlag(ItemIsSelectable, false);
  connect(parent, SIGNAL(xChanged()), this, SLOT(moved()));
  connect(parent, SIGNAL(yChanged()), this, SLOT(moved()));
}

Sink::~Sink()
{
}


Data::Formats Sink::allowedFormats() const
{
  return m_allowedFormats;
}

int Sink::type() const
{
  return Type;
}

Object* Sink::parentObject() const
{
  return m_parent;
}

Data* Sink::sourceData() const
{
  return (Data*)connectedSource()->parentObject();
}

Function* Sink::sourceFunction() const
{
  return (Function*)connectedSource()->parentObject();
}

bool Sink::isConnected() const
{
  return (m_connection != 0);
}

Source* Sink::connectedSource() const
{
  if (m_connection) {
    return m_connection->source();
  }
}

void Sink::setConnection(Connection* connection)
{
  if (!connection) {
    m_connection = connection;
    return;
  }
  
  Data::Format format = connection->source()->format();
  Q_ASSERT( allowedFormats().contains(format));
  
  m_connection = connection;
  m_connectionFormat = format;
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
  
  int location = m_allowedFormats.indexOf( m_connectionFormat);
  QPointF center(Size::inputHeight/2, Size::inputHeight/2);
  center += QPointF(location*(m_internalMargin + Size::inputHeight), 0);
  return center;
}

QRectF Sink::rect() const {
  qreal totalWidth = 0;
  foreach(Data::Format format, m_allowedFormats) {
    totalWidth += Size::inputHeight;
  }
  totalWidth += m_internalMargin*m_allowedFormats.length();

  QPointF topLeft = QPointF(0, 0);
  QPointF bottomRight = QPointF(totalWidth, Size::inputHeight);
  return QRectF(topLeft, bottomRight);
}

QRectF Sink::boundingRect() const
{
  return rect().adjusted(-1, -1, 1, 1);
}
void Sink::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  QPen p(Qt::black, 1, Qt::DotLine);
  painter->setPen(p);
  
  QPointF topLeft = QPointF(0, 0);
  
  foreach(Data::Format format, m_allowedFormats) {
    QPointF center = topLeft + QPointF(Size::inputHeight/2, Size::inputHeight/2);
    switch (format) {
      case Data::Value:
        painter->drawPath(triangle(center, Size::inputHeight, Size::inputHeight));
        break;
    
      case Data::DataBlock:
        painter->drawEllipse(center, Size::inputHeight/2, Size::inputHeight/2);
        break;
        
      default:
        qDebug() << "Unhandled datatype" << format;
        break;
    }
  topLeft = QPointF(topLeft.x() + Size::inputHeight + m_internalMargin, topLeft.y());
  }
}

void Sink::moved() {
  //qDebug() << "sink moved";
  if (m_connection) {
    m_connection->updateConnection();
  }
}

