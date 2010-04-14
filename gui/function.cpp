#include "function.h"
#include <QApplication>
#include <qfontmetrics.h>
#include <QPainter>

Function::Function(const QString& name, QGraphicsObject* parent)
  : QGraphicsObject(parent)
{
  m_hasOperation = false;
  m_operation = 0;
  m_name = name;
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
}
Operation* Function::operation() {
  return m_operation;
}

/// Input Goodness
void Function::addInputOption(Option option) {
  if (!m_inputOptions.contains(option)) {
    m_inputOptions.append(option);
  }
}
void Function::removeInputOption(Option option) {
  m_inputOptions.removeOne(option);
}
QList< Option > Function::inputsOptions() const {
  return m_inputOptions;
}
void Function::connectInput(int location, Input* input) {
  //TODO Check to see if the input is a valid option
  //TODO Check to see if there isn't already an input at that location
  m_inputs.append(InputConnection(location, input));
}
void Function::disconnectInput(int location, Input* input) {
  m_inputs.removeOne(InputConnection(location, input));
}
QList< InputConnection > Function::inputs() const {
  return m_inputs;
}


/// Output Goodness
void Function::addOutputOption(Option option) {
  if (!m_outputOptions.contains(option)) {
    m_outputOptions.append(option);
  }
}
void Function::removeOutputOption(Option option) {
  m_outputOptions.removeOne(option);
}
QList< Option > Function::outputOptions() const {
  return m_outputOptions;
}
void Function::connectOutput(int location, Output* output) {
  //TODO Check to see if the output is a valid option
  m_outputs.append(OutputConnection(location, output));
}
void Function::disconnectOutput(int location, Output* output) {
  m_outputs.removeOne(OutputConnection(location, output));
}
QList< OutputConnection > Function::outputs() const {
  return m_outputs;
}

/// Painting Goodness
QRectF Function::boundingRect() const
{
}
void Function::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
  // Draw Inputs
  painter->drawRoundedRect(inputsRect(), 5, 5);
  painter->drawText(inputsRect().topLeft()+QPointF(15, QApplication::fontMetrics().height()), "input1");
  painter->drawEllipse(inputsRect().topLeft()+QPointF(10, QApplication::fontMetrics().height()/2.0+4), 3, 3); 
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
    painter->drawEllipse(QPointF(0, 0), circleradius, circleradius);
  painter->restore();
  // Draw Outputs
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

}
