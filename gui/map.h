#ifndef CHESTNUT_MAP_H
#define CHESTNUT_MAP_H

#include "function.h"
#include "datautils.h"

class Map : public Function {
  public:
    Map(QGraphicsObject* parent = 0);
    virtual ~Map();
    
    enum { Type = ChestnutItemType::Map };
    int type() const;
    
    virtual ProgramStrings flatten() const;
  private:
};

#endif //CHESTNUT_MAP_H