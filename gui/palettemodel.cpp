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
  types << "application/x-chestnutpaletteheader";
  types << "application/x-chestnutpaletteitemoperator";
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
  

  if (data(indexes.at(0).parent()).toString() == "Operators") {
    mimeData->setData("application/x-chestnutpaletteitemoperator", encodedData);
  } else if (indexes.at(0).parent().isValid()) {
    mimeData->setData("application/x-chestnutpaletteitem", encodedData);
  } else {
    mimeData->setData("application/x-chestnutpaletteheader", encodedData);
  }
  return mimeData;
}
