#ifndef CHESTNUT_PALETTEMODEL_H
#define CHESTNUT_PALETTEMODEL_H

#include <QStandardItemModel>

/**
 * An internal chestnuct class to hold the items in the left pane
 */
class PaletteModel : public QStandardItemModel {
  public:
    PaletteModel(QObject* parent = 0);
    virtual QStringList mimeTypes() const;
    virtual QMimeData* mimeData(const QModelIndexList& indexes) const;
};


#endif //CHESTNUT_PALETTEMODEL_H