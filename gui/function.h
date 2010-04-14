#ifndef CHESTNUT_FUNCTION_H
#define CHESTNUT_FUNCTION_H

#include "type.h"

#include <QPair>
#include <QList>
#include <QGraphicsObject>

class Input;
class Output;

class Operation;

typedef QPair<int, Type> Option;
typedef QPair<int, Input*> InputConnection;
typedef QPair<int, Output*> OutputConnection;

class Function : public QGraphicsObject {
  public:
    Function(const QString& name, QGraphicsObject* parent = 0);
    virtual ~Function();
    
    bool hasOperation();
    Operation* operation();
    void setOperation(Operation *op);
    
    QList<Option> inputsOptions() const;
    void connectInput(int location, Input *input);
    void disconnectInput(int location, Input *input);
    QList<InputConnection>inputs() const;
    
    QList<Option> outputOptions() const;
    void connectOutput(int location, Output *output);
    void disconnectOutput(int location, Output *output);
    QList<OutputConnection> outputs() const;
    
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
  protected:
    void setHasOperation(bool hasOperation);
    
    void addInputOption(Option option);
    void removeInputOption(Option option);
    
    void addOutputOption(Option option);
    void removeOutputOption(Option option);
    
    QRectF inputsRect() const;
    QRectF internalRect() const;
    QRectF outputsRect() const;
    
  private:
    bool m_hasOperation;
    Operation *m_operation;
    QList<InputConnection> m_inputs;
    QList<OutputConnection> m_outputs;
    
    QList<Option> m_inputOptions;
    QList<Option> m_outputOptions;
    
    QString m_name;
    
};

#endif // CHESTNUT_FUNCTION_H