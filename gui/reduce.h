#ifndef CHESTNUT_REDUCE_H
#define CHESTNUT_REDUCE_H

#include "function.h"
#include "datautils.h"

/** This class represents the reduce() function in Chestnut code. It will
 * reduce an array of scalars into a single scalar. Currently this means
 * using the given operator (such as +, *) on successive pairs
 */
class Reduce : public Function {
  public:
    Reduce(QGraphicsObject* parent = 0);
    
    enum { Type = ChestnutItemType::Reduce };
    int type() const;
    
    virtual ProgramStrings flatten() const;
  private:
};

#endif //CHESTNUT_REDUCE_H
