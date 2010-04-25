#include "function.h"

#include "operation.h"
#include "sizes.h"
#include "sink.h"
#include "source.h"

#include <QApplication>
#include <qfontmetrics.h>
#include <QPainter>
#include <QDebug>

using namespace Chestnut;

Function::Function(const QString& name, QGraphicsObject* parent)
  : Object(parent)
{
  m_hasOperation = false;
  m_operation = 0;
  m_name = name;
}

Function::~Function() {}

bool Function::isFunction() const
{
  return true;
}

int Function::type() const
{
  return Type;
}

QString Function::name() const{
  return m_name;
}

/// Operation Goodness
void Function::setHasOperation(bool hasOperation) {
  m_hasOperation = hasOperation;
}
bool Function::hasOperation() {
  return m_hasOperation;
}
void Function::setOperation(Operation* op) {
  m_operation = op;
  m_operation->setPos(operationPos() - QPointF(Size::operatorRadius, Size::operatorRadius)); //TODO fix positioning
}
Operation* Function::operation() const {
  return m_operation;
}

QPointF Function::operationPos() const {
  return QPointF(0, 0);
}

void Function::addSink(Sink *sink) {
  if (m_sinks.contains(sink)) {
    return;
  }
  sink->setParent(this);

  QPointF topLeft;
  if (m_sinks.isEmpty()) {
    topLeft = inputsRect().topLeft() + QPointF(Size::inputsMarginX, Size::inputsMarginY);
  } else {
    Sink *furthestRight = m_sinks.last();
    QPointF topRight = mapFromItem(furthestRight, furthestRight->rect().topRight());
    topLeft = topRight  + QPointF(Size::outputsMarginX, 0);
  }
  sink->setPos(topLeft);
  m_sinks.append(sink);
}

void Function::addSource(Source *source) {
  if (m_sources.contains(source)) {
    return;
  }
  source->setParent(this);

  QPointF topLeft;
  if (m_sources.isEmpty()) {
    topLeft = outputsRect().topLeft() + QPointF(Size::outputsMarginX, Size::outputsMarginY);
  } else {
    Source *furthestRight = m_sources.last();
    QPointF topRight = mapFromItem(furthestRight, furthestRight->rect().topRight());
    topLeft = topRight  + QPointF(Size::outputsMarginX, 0);
  }
  source->setPos(topLeft);
  m_sources.append(source);
}

/// Painting Goodness
QRectF Function::boundingRect() const
{
  return inputsRect().united(internalRect()).united(outputsRect());
}
void Function::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  painter->save();
  if (isSelected()) {
    painter->setPen(QPen(Qt::DashLine));
  }
  painter->drawRoundedRect(inputsRect(), 5, 5);
  painter->drawRoundedRect(outputsRect(), 5, 5);
  painter->drawRect(internalRect());
  painter->restore();
  
  // Draw Internals
  qreal xpos = 0 - 0.5*QApplication::fontMetrics().width(m_name);
  qreal ypos = 0 - Size::operatorRadius - Size::operatorMargin;
  painter->drawText(xpos, ypos, m_name);
  if (hasOperation()) {
    painter->save();
      QPen pen(Qt::gray, 1, Qt::DotLine, Qt::RoundCap, Qt::RoundJoin);
      painter->setPen(pen);
      painter->drawEllipse(operationPos(), Size::operatorRadius, Size::operatorRadius);
    painter->restore();
  }
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
  qreal width = 2*xmargin + QApplication::fontMetrics().width(m_name);
  qreal height = 2*(Size::operatorRadius + Size::operatorMargin) + 2*ymargin + QApplication::fontMetrics().height();
  
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
