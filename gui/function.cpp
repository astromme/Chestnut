#include "function.h"

Function::Function(QGraphicsObject* parent)
  : QGraphicsObject(parent)
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





//     QList<Option> inputsOptions() const;
//     void connectInput(int location, Input *input);
//     void disconnectInput(int location, Input *input);
//     QList<InputConnection>inputs() const;
//     
//     QList<Option> outputOptions() const;
//     void connectOutput(int location, Output *output);
//     void disconnectOutput(int location, Output *output);
//     QList<OutputConnection> outputs() const;
//     
//   protected:
//     void setHasOperation(bool hasOperation);
//     
//     void addInputOption(Option option);
//     void removeInputOption(Option option);
//     
//     void addOutputOption(Option option);
//     void removeOutputOption(Option option);
