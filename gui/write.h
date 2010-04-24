#ifndef CHESTNUT_WRITE_H
#define CHESTNUT_WRITE_H

#include "function.h"
#include "datautils.h"

class Write : public Function {
  public:
    Write(QGraphicsObject* parent = 0);
    
    enum { Type = ChestnutItemType::Write };
    int type() const;
    
    QString filename() const;
    void setFilename(const QString &fname);    
    
    virtual ProgramStrings flatten() const;
  private:
    QString m_filename;
};

#endif //CHESTNUT_WRITE_H

