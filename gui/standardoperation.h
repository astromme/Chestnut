#ifndef CHESTNUT_STANDARDOPERATION_H
#define CHESTNUT_STANDARDOPERATION_H

#include "operation.h"

class StandardOperation : public Operation {
  public:
    enum StandardOp {
      Add = 0,
      Subtract = 1,
      Multiply = 3,
      Divide = 4
    };
    
    StandardOperation(StandardOp type, Object *parent = 0);
    ~StandardOperation();
    
    virtual QString flatten() const;
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
  private:
    StandardOp m_operationType;
    int m_radius;
};

#endif //CHESTNUT_STANDARDOPERATION_H