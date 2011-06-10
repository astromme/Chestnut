#include <QtCore/QtGlobal>

#if defined(WALNUT_LIBRARY)
#  define WALNUT_EXPORT Q_DECL_EXPORT
#else
#  define WALNUT_EXPORT Q_DECL_IMPORT
#endif
