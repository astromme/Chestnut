/*
 *   Copyright 2011 Andrew Stromme <astromme@chatonka.com>
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as
 *   published by the Free Software Foundation; either version 2, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details
 *
 *   You should have received a copy of the GNU Library General Public
 *   License along with this program; if not, write to the
 *   Free Software Foundation, Inc.,
 *   51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include "Array.h"
#include "ArrayAllocator.h"

#include <QFile>
#include <QRegExp>
#include <QDebug>
#include <QStringList>

#include <thrust/host_vector.h>

namespace Walnut {

template <typename T>
Array<T>::Array(thrust::device_vector<T> &vector, int width, int height, int depth) {
  data = thrust::raw_pointer_cast(&(vector[0]));
  this->width = width;
  this->height = height;
  this->depth = depth;
}

template <typename T>
T elementFromString(const QByteArray &string) {
    return string.trimmed().toDouble();
}

template <>
Color elementFromString(const QByteArray &string) {
    QStringList parts = QString(string).split(" ");
    Color c;
    c.red() = parts[0].toDouble();
    c.green() = parts[1].toDouble();
    c.blue() = parts[2].toDouble();
    c.opacity() = parts[3].toDouble();
    return c;
}

template <typename T>
QString stringFromElement(const T &element) {
    return QString::number(element);
}

template <>
QString stringFromElement(const Color &element) {
    return QString("%1 %2 %3 %4").arg(element.red())
                                 .arg(element.green())
                                 .arg(element.blue())
                                 .arg(element.opacity());
}

template <typename T>
bool Array<T>::readFromFile(const QString &fileName) {
    QFile file(fileName);

    if (!file.open(QFile::ReadOnly)) {
        qDebug() << QString("Error Reading %1: %2").arg(fileName, file.errorString()).toAscii();
        return false;
    }

    QByteArray firstLine = file.readLine().trimmed();
    QRegExp firstLineMatcher(QString("^(Int|Bool|Real)Array(1|2|3)d\\[([0-9]+), ([0-9]+)\\] = \\[$").replace(' ', "\\s+"));
    firstLineMatcher.indexIn(firstLine);

    qDebug() << firstLineMatcher.capturedTexts();

    QString type = firstLineMatcher.capturedTexts()[1];
    int dimension = firstLineMatcher.capturedTexts()[2].toInt();
    int width = firstLineMatcher.capturedTexts()[3].toInt();
    int height = firstLineMatcher.capturedTexts()[4].toInt();

    thrust::host_vector<T> host_data(width*height);

    int pos = 0;

    while (!file.atEnd()) {
        QByteArray line = file.readLine();
        foreach (QByteArray element, line.split(',')) {
            host_data[pos] = elementFromString<T>(element);
            pos++;
        }
    }

    // copy to device
    thrust::copy(host_data.begin(), host_data.end(), this->thrustPointer());
    return true;
}

template <typename T>
bool Array<T>::writeToFile(const QString &fileName) {
    QFile file(fileName);

    if (!file.open(QFile::WriteOnly)) {
        qDebug() << QString("Error Writing %1: %2").arg(fileName, file.errorString()).toAscii();
        return false;
    }

    thrust::host_vector<T> host_data(width*height*depth);
    this->copyTo(host_data);

    QStringList dimensions;
    dimensions.append(QString::number(width));
    dimensions.append(QString::number(height));


    QString header = QString("%1Array%2d[%3] = [\n").arg("Real", "2", dimensions.join(", "));
    file.write(header.toAscii());
    for (int y=0; y<height; y++) {
        QStringList row;
        for (int x=0; x<width; x++) {
            row.append((stringFromElement(host_data[calculateIndex(x, y, 0)])));
        }
        file.write(row.join(", ").toAscii());
        file.write("\n");
    }
    file.write("];\n");
    file.close();
    return true;
}

WALNUT_INIT_STRUCT(Array);

} // namespace Walnut

