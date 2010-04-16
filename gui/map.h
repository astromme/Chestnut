#ifndef CHESTNUT_MAP_H
#define CHESTNUT_MAP_H

#include "function.h"

class Map : public Function {
  public:
    Map(QGraphicsObject* parent = 0);
    
    virtual QList< QString > flatten() const;
  private:
};

#endif // CHESTNUT_MAP_H