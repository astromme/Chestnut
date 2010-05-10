#ifndef CHESTNUT_SORT_H
#define CHESTNUT_SORT_H

#include "function.h"
#include "datautils.h"

/**
 * This represents the sort() function in Chestnut code. It will
 * sort the array of numbers in ascending or decending order.
 * Currently only ascending order is supported.
 */
class Sort : public Function {
  public:
    Sort(QGraphicsObject* parent = 0);
    
    enum { Type = ChestnutItemType::Sort };
    int type() const;
    
    virtual ProgramStrings flatten() const;
  private:
};

#endif //CHESTNUT_SORT_H

