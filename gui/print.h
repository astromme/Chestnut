#ifndef CHESTNUT_PRINT_H
#define CHESTNUT_PRINT_H

#include "function.h"
#include "datautils.h"

/**
 * This class represents the print() function in Chestnut code.
 */
class Print : public Function {
  public:
    Print(QGraphicsObject* parent = 0);
    
    enum { Type = ChestnutItemType::Print };
    int type() const;
    
    virtual ProgramStrings flatten() const;
  private:
};

#endif //CHESTNUT_PRINT_H
