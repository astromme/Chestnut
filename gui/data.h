#ifndef CHESTNUT_DATA_H
#define CHESTNUT_DATA_H

#include <QList>
#include <QString>

#include "object.h"

class QValidator;

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
    virtual bool isInitialized() const;
    
    void setExpression(QString expr);
    
    QString expression() const;
    QString name() const;
    void setName(const QString &name);
    Format format() const;
    QString datatype() const;
    void setDatatype(const QString &datatype);
    
    /** returns a unique name for a temporary variable of category t */
    static QString tempData(Data::Format f);
    virtual ProgramStrings flatten() const {return ProgramStrings();} //TODO Fix
    
  protected:
    QValidator *m_nameValidator;
    
  private:
    Format m_format;
    QString m_datatype;
    QString m_name;
    QString m_expression;
};


#endif //CHESTNUT_DATA_H
