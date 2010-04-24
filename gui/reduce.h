#ifndef CHESTNUT_REDUCE_H
#define CHESTNUT_REDUCE_H

#include "function.h"
#include "datautils.h"

class Reduce : public Function {
  public:
    Reduce(QGraphicsObject* parent = 0);
    
    enum { Type = ChestnutItemType::Reduce };
    int type() const;
    
    virtual ProgramStrings flatten() const;
  private:
};

#endif //CHESTNUT_REDUCE_H
