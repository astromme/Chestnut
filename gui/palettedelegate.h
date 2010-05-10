#ifndef CHESTNUT_PALETTEDELEGATE_H
#define CHESTNUT_PALETTEDELEGATE_H

#include <QStyledItemDelegate>

/**
 * An internal chestnut class to modify how the left pane is visualized
 */
class PaletteDelegate : public QStyledItemDelegate {
  public:
    virtual void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const;
    virtual QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const;
};

#endif //CHESTNUT_PALETTEDELEGATE_H