#ifndef CHESTNUT_TYPES_H
#define CHESTNUT_TYPES_H

#include <QGraphicsItem>

// Internal types used for QGraphicsItem derived classes
// When adding a new type here ensure it doesn't conflict with any others
// i.e. make sure that the + xy; is unique for each.
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
  const int Print = QGraphicsItem::UserType + 24;
  const int Write = QGraphicsItem::UserType + 25;
  
  // Sinks/Sources
  const int Source = QGraphicsItem::UserType + 40;
  const int Sink = QGraphicsItem::UserType + 41;
  const int Connection = QGraphicsItem::UserType + 42;
}

#endif //CHESTNUT_TYPES_H
