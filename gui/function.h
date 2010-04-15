#ifndef CHESTNUT_FUNCTION_H
#define CHESTNUT_FUNCTION_H

#include "type.h"
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
    
    bool hasOperation();
    Operation* operation();
    void setOperation(Operation *op);
        
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
  protected:
    void setHasOperation(bool hasOperation);
        
    QRectF inputsRect() const;
    QRectF internalRect() const;
    QRectF outputsRect() const;
    
  private:
    bool m_hasOperation;
    Operation *m_operation;

    QString m_name;
    
};

#endif // CHESTNUT_FUNCTION_H