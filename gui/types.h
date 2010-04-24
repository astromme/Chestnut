#ifndef CHESTNUT_TYPES_H
#define CHESTNUT_TYPES_H

#include <QGraphicsItem>

namespace ChestnutItemType {
  
  // Data
  const int Data = QGraphicsItem::UserType + 10;
  const int Value = QGraphicsItem::UserType + 11;
  const int DataBlock = QGraphicsItem::UserType + 12;
  
  // Functions
  const int Function = QGraphicsItem::UserType + 20;
  const int Map = QGraphicsItem::UserType + 21;
  const int Sort = QGraphicsItem::UserType + 22;
  const int Reduce = QGraphicsItem::UserType + 23;
  
  // Sinks/Sources
  const int Source = QGraphicsItem::UserType + 30;
  const int Sink = QGraphicsItem::UserType + 31;
}

#endif //CHESTNUT_TYPES_H
