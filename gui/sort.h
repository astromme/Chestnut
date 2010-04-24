#ifndef CHESTNUT_SORT_H
#define CHESTNUT_SORT_H

#include "function.h"
#include "datautils.h"

class Sort : public Function {
  public:
    Sort(QGraphicsObject* parent = 0);
    
    enum { Type = ChestnutItemType::Sort };
    int type() const;
    
    virtual ProgramStrings flatten() const;
  private:
};

#endif //CHESTNUT_SORT_H

