#ifndef CHESTNUT_DATABLOCK_H
#define CHESTNUT_DATABLOCK_H

#include <QGraphicsObject>

#include "data.h"

namespace Ui {
  class DataBlockProperties;
}

// assumed to be 2d
class DataBlock : public Data {
  public:
    DataBlock(const QString &name, const QString &datatype, int rows, int columns);
    virtual ~DataBlock();
    
    enum { Type = ChestnutItemType::DataBlock };
    int type() const;
    
    int rows() const;
    int columns() const;
    
    virtual ProgramStrings flatten() const;
    
    virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event);
    
    QRectF rect() const;
    virtual QRectF boundingRect() const;
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  private:
    int m_dimension;
    int m_rows;
    int m_columns;
    Ui::DataBlockProperties* m_ui;
};

#endif //CHESTNUT_DATABLOCK_H