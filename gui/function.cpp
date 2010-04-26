#include "function.h"

#include "operation.h"
#include "sizes.h"
#include "sink.h"
#include "source.h"

#include <QApplication>
#include <qfontmetrics.h>
#include <QPainter>
#include <QDebug>
#include <QGraphicsSceneMouseEvent>
#include "standardoperation.h"
#include <QMimeData>

using namespace Chestnut;

Function::Function(const QString& name, QGraphicsObject* parent)
  : Object(parent)
{
  m_hasOperation = false;
  m_hasInputs = false;
  m_hasOutputs = false;
  m_operation = 0;
  m_name = name;
}

Function::~Function()
{
}

bool Function::isFunction() const
{
  return true;
}

bool Function::isFullyConnected() const
{
  if (m_hasOperation && (m_operation == 0)) {
    return false;
  }
  foreach(Sink *sink, sinks()) {
    if (!sink->isConnected()) {
      return false;
    }
  }
  foreach(Source *source, sources()) {
    if (source->connectedSinks().size() == 0) {
      return false;
    } 
  }
  return true;
}

int Function::type() const
{
  return Type;
}

QString Function::name() const{
  return m_name;
}

bool Function::hasInputs() const
{
  return m_hasInputs;
}
void Function::setHasInputs(bool hasInputs)
{
  m_hasInputs = hasInputs;
}
bool Function::hasOutputs() const
{
  return m_hasOutputs;
}
void Function::setHasOutputs(bool hasOutputs)
{
  m_hasOutputs = hasOutputs;
}

/// Operation Goodness
void Function::setHasOperation(bool hasOperation) {
  m_hasOperation = hasOperation;
  setAcceptDrops(hasOperation);
}
bool Function::hasOperation() const {
  return m_hasOperation;
}
void Function::setOperation(Operation* op) {
  if (m_operation) {
    delete m_operation;
  }
  m_operation = op;
  m_operation->setPos(operationPos());
}
Operation* Function::operation() const {
  return m_operation;
}

QPointF Function::operationPos() const {
  QRectF rect = internalRect().adjusted(0, inputsRect().height(), 0, -outputsRect().height());
  qreal xpos = rect.right() - Size::operatorMargin - Size::operatorRadius*2 - 6;
  qreal ypos = rect.top() + rect.height()/2 - Size::operatorRadius;
  
  return QPointF(xpos, ypos);
}

void Function::addSink(Sink *sink) {
  if (m_sinks.contains(sink)) {
    return;
  }
  sink->setParent(this);

  QPointF topLeft;
  if (m_sinks.isEmpty()) {
    topLeft = inputsRect().topLeft() + QPointF(0, inputsRect().height()/2);
    topLeft += QPointF(Size::inputsTextWidth(), -Size::inputRadius);
    topLeft += QPointF(Size::inputsMarginX, 0);
  } else {
    Sink *furthestRight = m_sinks.last();
    QPointF topRight = mapFromItem(furthestRight, furthestRight->rect().topRight());
    topLeft = topRight  + QPointF(2*Size::inputsMarginX, 0);
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
    topLeft = outputsRect().topLeft() + QPointF(0, outputsRect().height()/2);
    topLeft += QPointF(Size::outputTextWidth(), -Size::outputRadius);
    topLeft += QPointF(Size::outputsMarginX, 0);
  } else {
    Source *furthestRight = m_sources.last();
    QPointF topRight = mapFromItem(furthestRight, furthestRight->rect().topRight());
    topLeft = topRight  + QPointF(2*Size::outputsMarginX, 0);
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
  painter->drawRoundedRect(internalRect(), 5, 5);
  painter->restore();
    
  painter->save();
  painter->setPen(Qt::gray);
  painter->drawLine(inputsRect().bottomLeft(), inputsRect().bottomRight());
  qreal sinkSeparator = Size::inputsTextWidth();
  foreach(Sink *sink, sinks()) {
    if (sink != sinks().last()) {
      sinkSeparator += sink->rect().width() + 2*Size::inputsMarginX;
      painter->drawLine(sinkSeparator, inputsRect().top()+4, sinkSeparator, inputsRect().bottom()-4);
    }
  }
  
  if (sources().size() > 0) {
    painter->drawLine(outputsRect().topLeft(), outputsRect().topRight());
    qreal sourceSeparator = Size::inputsTextWidth();
    foreach(Source *source, sources()) {
      if (source != sources().last()) {
        sourceSeparator += source->rect().width() + 2*Size::inputsMarginX;
        painter->drawLine(sourceSeparator, outputsRect().top()+4, sourceSeparator, outputsRect().bottom()-4);
      }
    }
  }
  
  painter->restore();
  painter->drawText(inputsRect(), Qt::AlignLeft | Qt::AlignVCenter, Size::inputsText);
  painter->drawText(outputsRect(), Qt::AlignLeft | Qt::AlignVCenter, Size::outputsText);
  
  // Draw Internals
  if (hasOperation()) {
    painter->drawText(internalRect().adjusted(10, inputsRect().height(), 0, -outputsRect().height()),
                      Qt::AlignVCenter | Qt::AlignLeft, m_name + "()");
  } else {
    painter->drawText(internalRect().adjusted(0, inputsRect().height(), 0, -outputsRect().height()),
                      Qt::AlignCenter, m_name + "()");
  }
  if (hasOperation()) {
    painter->save();
      QPen pen(Qt::gray, 1, Qt::DotLine, Qt::RoundCap, Qt::RoundJoin);
      painter->setPen(pen);
      painter->drawEllipse(operationPos()+QPointF(Size::operatorRadius, Size::operatorRadius),
                           Size::operatorRadius, Size::operatorRadius);
    painter->restore();
  }
}

QRectF Function::inputsRect() const
{
  if (!hasInputs()) {
    return QRectF();
  }
  
  qreal margin = 2;
  QPointF topLeft = QPointF(0, 0);
  qreal height = QApplication::fontMetrics().height() + 2*margin;
  qreal width = Size::inputsTextWidth() + margin;
  
  foreach(Sink *sink, sinks()) {
    width += sink->rect().width();
    width += margin*2;
  }
  
  qreal outputsWidth = Size::outputTextWidth() + margin;
  
  foreach(Source *source, sources()) {
    outputsWidth += source->rect().width();
    outputsWidth += margin*2;
  } //TODO: don't do calculation in outputsRect() as well
  
  qreal xmargin = 5;
  qreal midRectWidth = (2*hasOperation()+4)*xmargin + QApplication::fontMetrics().width(m_name) + 2*(Size::operatorRadius + Size::operatorMargin);
  
  width = qMax(width, outputsWidth);
  width = qMax(width, midRectWidth);
  
  return QRectF(topLeft, QSizeF(width, height));
}
QRectF Function::internalRect() const
{
  qreal xmargin = 5;
  qreal ymargin = 5;
  qreal width = qMax(inputsRect().width(), outputsRect().width());
  //width += 10;
  width = qMax(width, (2*hasOperation()+4)*xmargin + QApplication::fontMetrics().width(m_name) + 2*(Size::operatorRadius + Size::operatorMargin));
  
  qreal height = inputsRect().height() + outputsRect().height() + 2*ymargin;
  height += qMax(2*ymargin + QApplication::fontMetrics().height(), 2*(Size::operatorRadius + Size::operatorMargin));
  
  return QRectF(QPointF(0, 0), QSizeF(width, height));
}
QRectF Function::outputsRect() const
{
  if (!hasOutputs()) {
    return QRectF();
  }
  
  qreal margin = 2;
  
  qreal ypos = inputsRect().height();
  ypos += qMax((qreal)10 + QApplication::fontMetrics().height(), 2*(Size::operatorRadius + Size::operatorMargin));
  QPointF topLeft = QPointF(0, ypos+10); // 10 == more margin
  
  qreal height = QApplication::fontMetrics().height() + 2*margin;
  qreal width = Size::outputTextWidth() + margin;
  
  foreach(Source *source, sources()) {
    width += source->rect().width();
    width += margin*2;
  }
  
  width = qMax(width, inputsRect().width());
  
  return QRectF(topLeft, QSizeF(width, height));
}

void Function::dragEnterEvent(QGraphicsSceneDragDropEvent* event)
{
  //qDebug() << "Function DragNDrop Enter";
  //qDebug() << event->mimeData()->formats();
  if (event->mimeData()->hasFormat("application/x-chestnutpaletteitemoperator")) {
    //qDebug() << "accepting";
    event->setProposedAction(Qt::CopyAction);
    event->accept();
    return;
  }
  event->setAccepted(false);
}
void Function::dragMoveEvent(QGraphicsSceneDragDropEvent* event)
{
  if (event->mimeData()->hasFormat("application/x-chestnutpaletteitemoperator")) {
    event->setProposedAction(Qt::CopyAction);
    event->accept();
    return;
  }
  event->setAccepted(false);
}
void Function::dropEvent(QGraphicsSceneDragDropEvent* event)
{
   //qDebug() << "Function DragNDrop Drop";
  if (event->mimeData()->hasFormat("application/x-chestnutpaletteitemoperator")) {
    QByteArray encodedData = event->mimeData()->data("application/x-chestnutpaletteitemoperator");
    QDataStream stream(&encodedData, QIODevice::ReadOnly);
    QStringList newItems;

    while (!stream.atEnd()) {
        QString text;
        stream >> text;
        newItems << text;
    }
    QString droppedItem = newItems.first();
    
    Operation *newItem = 0;
        
    //Operators
    if (droppedItem == "Add") { newItem = new StandardOperation(StandardOperation::Add, this); }
    else if (droppedItem == "Subtract") { newItem = new StandardOperation(StandardOperation::Subtract, this); }
    else if (droppedItem == "Multiply") { newItem = new StandardOperation(StandardOperation::Multiply, this); }
    else if (droppedItem == "Divide") { newItem = new StandardOperation(StandardOperation::Divide, this); }
  
    else {
      qDebug() << "Unknown Item: " << droppedItem;
      return;
    }
    
    setOperation(newItem);
    return;
  }
}

