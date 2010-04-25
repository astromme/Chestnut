#include "palettemodel.h"

#include <QMimeData>
#include <QDebug>

PaletteModel::PaletteModel(QObject* parent)
  : QStandardItemModel(parent)
{

}

QStringList PaletteModel::mimeTypes() const
{
  QStringList types;
  types << "application/x-chestnutpaletteitem";
  return types;
}

QMimeData* PaletteModel::mimeData(const QModelIndexList& indexes) const
{
  QMimeData *mimeData = new QMimeData();
  QByteArray encodedData;

  QDataStream stream(&encodedData, QIODevice::WriteOnly);

  foreach (QModelIndex index, indexes) {
    if (index.isValid()) {
      qDebug() << "data" << data(index, Qt::DisplayRole);
      QString text = data(index, Qt::DisplayRole).toString();
      stream << text;
    }
  }

  mimeData->setData("application/x-chestnutpaletteitem", encodedData);
  return mimeData;
}
