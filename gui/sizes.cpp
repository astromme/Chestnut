#include "sizes.h"

#include <QApplication>
#include <QFontMetrics>

qreal Chestnut::Size::inputsTextWidth()
{
  return QApplication::fontMetrics().width(inputsText);
}

qreal Chestnut::Size::outputTextWidth()
{
  return QApplication::fontMetrics().width(outputsText);
}
