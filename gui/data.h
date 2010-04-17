#ifndef CHESTNUT_DATA_H
#define CHESTNUT_DATA_H

#include <QList>
#include <QString>

#include "object.h"

class Data : public Object {
  public:
    enum Type {
      Value = 0x0,
      DataBlock = 0x1
    };
    
    typedef QList<Type> Types;
    
    Data(const QString &name);
    virtual ~Data();
    
    virtual QStringList flatten() const {return QStringList();} //TODO Fix
};


#endif //CHESTNUT_DATA_H