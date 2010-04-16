#ifndef CHESTNUT_OPERATION_H
#define CHESTNUT_OPERATION_H

#include <QGraphicsObject>

class Object;

class Operation : public QGraphicsObject {
  public:
    Operation(Object *parent);
    virtual ~Operation();
    
    virtual QString flatten() const = 0;
};

#endif //CHESTNUT_OPERATION_H