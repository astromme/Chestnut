#ifndef CHESTNUT_MAP_H
#define CHESTNUT_MAP_H

#include "function.h"
#include "datautils.h"

/**
 * This class represents the map() function in Chestnut code. It
 * performs the same operation on each element of the given DataBlock
 */
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