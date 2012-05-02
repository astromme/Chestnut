{% block includes -%}
// OpenGL stuff on Linux forces QApplication to come first
#include <QApplication> 

// Walnut Library includes that allow for a much closer mapping of
// Chestnut onto C++/CUDA/Thrust
#include <walnut/ArrayAllocator.h>
#include <walnut/Array.h>
#include <walnut/HostFunctions.h>
#include <walnut/UtilityFunctors.h>
#include <walnut/FunctionIterator.h>
#include <walnut/DisplayWindow.h>
#include <walnut/ColorKernels.h>
#include <walnut/Sizes.h>
#include <walnut/Points.h>
#include <walnut/Windows.h>
#include <walnut/Color.h>

// Most of thrust is hidden behind Walnut but in this case we need to
// include the sorting support explicitly
#include <thrust/sort.h>

#include <limits.h>
{%- endblock %}

{% block using -%}
// We don't care about namespace collisions so much here because we're
// an end application, not a library
using namespace Walnut;
{%- endblock %}

{% block globals -%}
// This allocator manages the arrays throughout the runtime of this
// program. It knows when to reuse existing arrays and when to create
// new ones.
ArrayAllocator _allocator;

// Global timer for GPU kernel calls
unsigned int _host_timer;
{%- endblock %}

// Structures //
{%- for structure in symbolTable.structures %}
{{structure}}
{%- endfor %}
// END Structures //

// Declarations //
{%- for declaration in declarations %}
{{declaration}}
{%- endfor %}
// END Declarations //

// Parallel Contexts //
{%- for context in symbolTable.parallelContexts %}
{{context}}
{%- endfor %}
// END Parallel Contexts //

// Functions //
{%- for function in functions %}
{{function}}
{%- endfor %}
// END Functions //

{% block main -%}
int main(int argc, char* argv[]) {

  {%- if symbolTable.displayWindows|length > 0 %}

  // QApp needed for DisplayWindows
  QApplication _app(argc, argv);
  {%- endif %}

  srand(NULL);
  _allocator = ArrayAllocator();

  {% if symbolTable.displayWindows|length -%}
  // Create GUI Windows as needed
  {% for s in symbolTable.displayWindows -%}
  DisplayWindow _{{s.name}}_display(QSize({{s.width}}, {{s.height}}));
  _{{s.name}}_display.show();
  _{{s.name}}_display.setWindowTitle("{{s.title}}");
  {%- endfor %}{%- endif %}

  // Declarations
  // TODO

  {% block main_statments -%}
  {%- for line in main_statements %}
  {{line}}
  {%- endfor %}
  {%- endblock %}

  {% if symbolTable.displayWindows|length > 0 -%}
  // run until user closes window or quits
  return _app.exec();
  {%- else %}
  // Synchronize the threads because apparently kernels get aborted if we
  // return 0 without synchronizing?
  cudaThreadSynchronize();
  // quit once done with computations
  return 0;
  {%- endif %}
}
{%- endblock -%}
