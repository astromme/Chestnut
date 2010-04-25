#include "palettedelegate.h"
#include <QPainter>

void PaletteDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
  //painter->drawRect(QRect(option.rect.adjusted(1, 1, -1, -1)));
  QStyledItemDelegate::paint(painter, option, index);
}

QSize PaletteDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
  //return QSize(40, 40);
  return QStyledItemDelegate::sizeHint(option, index);
}
