#ifndef CHESTNUT_DATA_H
#define CHESTNUT_DATA_H

#include <QObject>

class Data {
  public:
    enum Type {
      Value = 0x0,
      DataBlock = 0x1
    };
    
    typedef QList<Type> Types;
    //Q_DECLARE_FLAGS(Types, Type)
};

//Q_DECLARE_OPERATORS_FOR_FLAGS(Data::Types)

#endif //CHESTNUT_DATA_H