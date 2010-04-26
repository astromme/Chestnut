#ifndef CHESTNUT_FUNCTION_H
#define CHESTNUT_FUNCTION_H

#include "object.h"

#include <QPair>
#include <QList>
#include <QGraphicsObject>

class Input;
class Output;

class Operation;

class Function : public Object {
  public:
    Function(const QString& name, QGraphicsObject* parent = 0);
    virtual ~Function();
    
    enum { Type = ChestnutItemType::Function };
    int type() const;
    
    virtual bool isFunction() const;
    
    /** Checks to see if this function has the necessary sources/sinks/operator
      * @returns true if everything is as it should be
      * @returns false if there is a problem.
      */
    bool isFullyConnected() const;
    
    //virtual QStringList flatten() const = 0;
    QString name() const;
    
    bool hasOperation();
    Operation* operation() const;
    QPointF operationPos() const;
    void setOperation(Operation *op); /**< Adds and sets the position of the given operation */
        
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
  protected:
    void setHasOperation(bool hasOperation);
    void addSource(Source *source); /**< Adds and sets the position of the given source */
    void addSink(Sink *sink); /**< Adds and sets the position of the given sink */
    
    virtual void dragEnterEvent(QGraphicsSceneDragDropEvent* event);
    virtual void dragMoveEvent(QGraphicsSceneDragDropEvent* event);
    virtual void dropEvent(QGraphicsSceneDragDropEvent* event);
        
    QRectF inputsRect() const;
    QRectF internalRect() const;
    QRectF outputsRect() const;
    
  protected:
    bool m_hasOperation;
    Operation *m_operation;
    QString m_name;
    
};

#endif // CHESTNUT_FUNCTION_H
