#ifndef CHESTNUT_SIZES_H
#define CHESTNUT_SIZES_H

#include "qglobal.h"
#include <QString>

namespace Chestnut {
  // Some simple sizes to allow consistent GUI painting
  namespace Size {
    static qreal inputsMarginX = 2;
    static qreal inputsMarginY = 2;
    static qreal inputRadius = 4;
    static qreal inputHeight = 2*inputRadius;
    static qreal inputWidth = 2*inputRadius;
    static QString inputsText = " Inputs ";
    qreal inputsTextWidth();

    static qreal outputsMarginX = 2;
    static qreal outputsMarginY = 2;
    static qreal outputRadius = 4;
    static qreal outputHeight = 2*outputRadius;
    static qreal outputWidth = 2*outputRadius;
    static QString outputsText = " Outputs ";
    qreal outputTextWidth();

    static qreal operatorRadius = 10;
    static qreal operatorMargin = 3;

    static qreal valueWidth = 60;
    static qreal valueHeight = 60;

    static qreal dataBlockWidth = 100;
    static qreal dataBlockHeight = 45;
  }
}

#endif //CHESTNUT_SIZES_H
