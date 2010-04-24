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
    
    QString name() const;
    Type dataType() const;
    
    /** returns a unique name for a temporary variable of type t */
    static QString tempData(Type t);
    virtual ProgramStrings flatten() const {return ProgramStrings();} //TODO Fix
  private:
    QString m_name;
};


#endif //CHESTNUT_DATA_H
