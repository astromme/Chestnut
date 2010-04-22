#include "standardoperation.h"

#include "sizes.h"

#include <QPainter>
#include <QDebug>

using namespace Chestnut;

StandardOperation::StandardOperation(StandardOperation::StandardOp type, Object* parent)
  : Operation("dummy", parent)
{
  m_operationType = type;
  switch (m_operationType) {
  case Add:
    m_name = "+";
    break;
  case Subtract:
    m_name = "-";
    break;
  case Multiply:
    m_name = "*";
    break;
  case Divide:
    m_name = "/";
    break;
  default:
    qDebug() << "Warning, operation type unknown";
    break;
  }
}

StandardOperation::~StandardOperation()
{

}

QRectF StandardOperation::boundingRect() const
{
  QPointF topLeft(0, 0);
  QPointF bottomRight = QPointF(2*Size::operatorRadius, 2*Size::operatorRadius) + QPointF(1, 1);
  return QRectF(topLeft, bottomRight);
}
void StandardOperation::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  QPointF center = QPointF(Size::operatorRadius, Size::operatorRadius);
  painter->drawEllipse(center, Size::operatorRadius, Size::operatorRadius);
  
  switch (m_operationType) {
    case Add:
      painter->drawText(boundingRect(), Qt::AlignCenter, "+");
      break;
    case Subtract:
      painter->drawText(boundingRect(), Qt::AlignCenter, "-");
      break;
    case Multiply:
      painter->drawText(boundingRect(), Qt::AlignCenter, "*");
      break;
    case Divide:
      painter->drawText(boundingRect(), Qt::AlignCenter, "/");
      break;
    default:
      qDebug() << "Warning, operation type unknown";
      break;
  }
}
