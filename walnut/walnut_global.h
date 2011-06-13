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

#include <QtCore/QtGlobal>

#if defined(WALNUT_LIBRARY)
#  define WALNUT_EXPORT Q_DECL_EXPORT
#else
#  define WALNUT_EXPORT Q_DECL_IMPORT
#endif

#define WALNUT_INIT_STRUCT_WITH_TYPE(name, datatype) template struct name<datatype>
#define WALNUT_INIT_STRUCT(name) WALNUT_INIT_STRUCT_WITH_TYPE(name, int); \
                                 WALNUT_INIT_STRUCT_WITH_TYPE(name, char); \
                                 WALNUT_INIT_STRUCT_WITH_TYPE(name, float); \
                                 WALNUT_INIT_STRUCT_WITH_TYPE(name, uchar4)
