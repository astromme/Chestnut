#ifndef CHESTNUT_FUNCTION_H
#define CHESTNUT_FUNCTION_H

#include "object.h"

#include <QPair>
#include <QList>
#include <QGraphicsObject>

class Input;
class Output;

class Operation;

/**
 * This is the base class for all Chestnut functions. It is not meant to
 * be used directly but rather should be subclassed into something that
 * initializes the necessary sinks and implements the correct flatten()
 * function.
 */
class Function : public Object {
  public:
    Function(const QString& name, QGraphicsObject* parent = 0);
    virtual ~Function();
    
    enum { Type = ChestnutItemType::Function };
    int type() const;
    
    virtual bool isFunction() const;
    
    /** 
     * Checks to see if this function has the necessary sources/sinks/operator
     *
     * @return @p true if everything is as it should be,
     *         @p false if there is a problem
     */
    bool isFullyConnected() const;
    
    //virtual QStringList flatten() const = 0;
    QString name() const;
    
    bool hasInputs() const;
    bool hasOutputs() const;
    
    bool hasOperation() const;
    Operation* operation() const;
    QPointF operationPos() const;
    void setOperation(Operation *op); /**< Adds and sets the position of the given operation */
        
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
  protected:
    void setHasOperation(bool hasOperation);
    void setHasInputs(bool hasInputs);
    void setHasOutputs(bool hasOutputs);
    
    void addSource(Source *source); /**< Adds and sets the position of the given source */
    void addSink(Sink *sink); /**< Adds and sets the position of the given sink */
    
    virtual void dragEnterEvent(QGraphicsSceneDragDropEvent* event);
    virtual void dragMoveEvent(QGraphicsSceneDragDropEvent* event);
    virtual void dropEvent(QGraphicsSceneDragDropEvent* event);
        
    QRectF inputsRect() const;
    QRectF internalRect() const;
    QRectF outputsRect() const;
    
  protected:
    QString m_name;
    
  private:
    bool m_hasOperation;
    Operation *m_operation;
    bool m_hasInputs;
    bool m_hasOutputs;
    
};

#endif // CHESTNUT_FUNCTION_H
