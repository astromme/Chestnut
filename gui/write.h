#ifndef CHESTNUT_WRITE_H
#define CHESTNUT_WRITE_H

#include "function.h"
#include "datautils.h"

/**
 * This object represents a write() function in Chestnut. It provides the
 * painting and chestnut code to implement writing a datablock or a value
 * out to a file.
 */
class Write : public Function {
  public:
    Write(QGraphicsObject* parent = 0);
    
    enum { Type = ChestnutItemType::Write };
    int type() const;
    
    /** 
     * @return the name of the file that will be written to
     */
    QString filename() const;
    /**
     * Set the name of the file that will be written to
     */
    void setFilename(const QString &fname);    
    
    virtual ProgramStrings flatten() const;
  private:
    QString m_filename;
};

#endif //CHESTNUT_WRITE_H

