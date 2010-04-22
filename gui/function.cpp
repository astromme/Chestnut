#include "function.h"

#include "operation.h"
#include "sizes.h"

#include <QApplication>
#include <qfontmetrics.h>
#include <QPainter>

Function::Function(const QString& name, QGraphicsObject* parent)
  : Object(name, parent)
{
  m_hasOperation = false;
  m_operation = 0;
}

Function::~Function() {}

/// Operation Goodness
void Function::setHasOperation(bool hasOperation) {
  m_hasOperation = hasOperation;
}
bool Function::hasOperation() {
  return m_hasOperation;
}
void Function::setOperation(Operation* op) {
  m_operation = op;
  m_operation->setPos(operationPos() - QPointF(Chestnut::operatorRadius, Chestnut::operatorRadius)); //TODO fix positioning
}
Operation* Function::operation() const {
  return m_operation;
}

QPointF Function::operationPos() const {
  return QPointF(0, 0);
}

/// Painting Goodness
QRectF Function::boundingRect() const
{
  return inputsRect().united(internalRect()).united(outputsRect());
}
void Function::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  // Draw Inputs
  painter->drawRoundedRect(inputsRect(), 5, 5);
  // Draw Internal Rect
  qreal circleradius = 10;
  qreal circlemargin = 3;
  painter->drawRect(internalRect());
  qreal xpos = 0 - 0.5*QApplication::fontMetrics().width(m_name);
  qreal ypos = 0 - circleradius - circlemargin;
  painter->drawText(xpos, ypos, m_name);
  painter->save();
    QPen pen(Qt::gray, 1, Qt::DotLine, Qt::RoundCap, Qt::RoundJoin);
    painter->setPen(pen);
    painter->drawEllipse(operationPos(), circleradius, circleradius);
  painter->restore();
  // Draw Outputs
  painter->drawRoundedRect(outputsRect(), 5, 5);
}

QRectF Function::inputsRect() const
{
  qreal margin = 2;
  QRectF internal = internalRect();
  QPointF bottomRight = internal.topRight();
  qreal width = internal.width();
  qreal height = QApplication::fontMetrics().height() + 2*margin;
  return QRectF(QPointF(bottomRight - QPointF(width, height)), bottomRight);
}
QRectF Function::internalRect() const
{
  qreal xmargin = 5;
  qreal ymargin = 5;
  qreal circleradius = 10;
  qreal circlemargin = 3;
  qreal width = 2*xmargin + QApplication::fontMetrics().width(m_name);
  qreal height = 2*(circleradius + circlemargin) + 2*ymargin + QApplication::fontMetrics().height();
  
  qreal xpos = 0 - width/2;
  qreal ypos = 0 - height/2;
  
  QRectF rect(QPointF(xpos, ypos), QSizeF(width, height));
  return rect;
}
QRectF Function::outputsRect() const
{
  qreal margin = 2;
  QRectF internal = internalRect();
  QPointF topLeft = internal.bottomLeft();
  qreal width = internal.width();
  qreal height = QApplication::fontMetrics().height() + 2*margin;
  return QRectF(QPointF(topLeft), topLeft + QPointF(width, height));
}
