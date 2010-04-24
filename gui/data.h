#ifndef CHESTNUT_DATA_H
#define CHESTNUT_DATA_H

#include <QList>
#include <QString>

#include "object.h"

class Data : public Object {
  public:
    enum Format {
      Value = 0x0,
      DataBlock = 0x1
    };
    
    typedef QList<Format> Formats;
    
    Data(const QString &name, Format format, const QString &datatype);
    virtual ~Data();
    
    enum { Type = ChestnutItemType::Map };
    int type() const;
    
    virtual bool isData() const;
    
    QString name() const;
    Format format() const;
    QString datatype() const;
    
    /** returns a unique name for a temporary variable of category t */
    static QString tempData(Data::Format f);
    virtual ProgramStrings flatten() const {return ProgramStrings();} //TODO Fix
    
  private:
    Format m_format;
    QString m_datatype;
    QString m_name;
};


#endif //CHESTNUT_DATA_H
