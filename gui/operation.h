#ifndef CHESTNUT_OPERATION_H
#define CHESTNUT_OPERATION_H

#include <QGraphicsObject>

class Object;

class Operation : public QGraphicsObject {
  public:
    Operation(Object *parent);
    
};

#endif //CHESTNUT_OPERATION_H