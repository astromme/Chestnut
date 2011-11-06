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

#include "HostFunctions.h"
#include "Array.h"

#include <QStringList>

namespace Walnut {

template <typename T>
std::string stdStringFromElement(const T &number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

template <typename T>
T elementFromString(const QString &string) {
    return string.trimmed().toDouble();
}

template <>
Color elementFromString(const QString &string) {
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
void printArray2D(const thrust::host_vector<T> &vector, int width, int height) {
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      int i = y*width + x;
      std::cout << ((vector[i] == 0) ? "." : stdStringFromElement(vector[i])) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void printArray(const thrust::host_vector<T> &vector, const Size3d &size) {
    for (int z=0; z<size.depth(); z++) {
        for (int y=0; y<size.height(); y++) {
            for (int x=0; x<size.width(); x++) {
                int i = z*(size.width()*size.height()) + y*size.width() + x;
                std::cout << ((vector[i] == 0) ? "." : stdStringFromElement(vector[i])) << " ";
             }
             std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

#define initPrintFunctionsWithType(T) \
template std::string stdStringFromElement(const T &number); \
template T elementFromString(const QString &string); \
template QString stringFromElement(const T &element); \
template void printArray2D(const thrust::host_vector<T> &vector, int width, int height); \
template void printArray(const thrust::host_vector<T> &vector, const Size3d &size);

initPrintFunctionsWithType(int);
initPrintFunctionsWithType(char);
initPrintFunctionsWithType(float);

} // namespace Walnut
