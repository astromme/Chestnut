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
#include "HostFunctions.h"

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
bool Array<T>::readFromFile(const QString &fileName) {
    QFile file(fileName);

    if (!file.open(QFile::ReadOnly)) {
        qDebug() << QString("Error Reading %1: %2").arg(fileName, file.errorString()).toAscii();
        return false;
    }

    QByteArray firstLine = file.readLine().trimmed();
    QRegExp firstLineMatcher(QString("^\\s*(Int|Bool|Real)Array(1|2|3)d\\[width=([0-9]+), height=([0-9]+)\\]\\s*$").replace(' ', "\\s+"));
    firstLineMatcher.indexIn(firstLine);

    qDebug() << firstLineMatcher.capturedTexts();

    QString type = firstLineMatcher.capturedTexts()[1];
    int dimension = firstLineMatcher.capturedTexts()[2].toInt();
    int width = firstLineMatcher.capturedTexts()[3].toInt();
    int height = firstLineMatcher.capturedTexts()[4].toInt();

    thrust::host_vector<T> host_data(width*height*depth);

    int pos = 0;

    QString window;
    while (!file.atEnd()) {
        window += file.read(1024);
        QStringList elements = window.split(',');
        if (elements.size() == 0) {
            continue;
        }

        for (int i=0; i<elements.size()-1; i++) {
            if (pos >= width*height*depth) {
                file.close();
                qDebug() << "Array full with data left in file.";
                break;
            }

            host_data[pos] = elementFromString<T>(elements[i]);
            pos++;
        }

        window = elements.last();
    }

    // get that last element
    host_data[pos] = elementFromString<T>(window);

    file.close();

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

    QString header = QString("%1Array%2d[width=%3, height=%4]\n").arg("Real", "2", QString::number(width), QString::number(height));
    file.write(header.toAscii());
    for (int z=0; z<depth; z++) {
        for (int y=0; y<height; y++) {
            QStringList row;
            for (int x=0; x<width; x++) {
                row.append((stringFromElement(host_data[calculateIndex(x, y, z)])));
            }
            file.write(row.join(", ").toAscii());
            if (y < height-1) {
                file.write(",");
            }
            file.write("\n");
        }
        file.write("\n");
    }
    file.close();
    return true;
}

WALNUT_INIT_STRUCT(Array);

} // namespace Walnut

