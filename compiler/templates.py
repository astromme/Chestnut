from symboltable import scalar_types, data_types
import numpy
from pycuda.gpuarray import vec as types

#helper to do nice code indenting.. duplicated in nodes, yuck
def indent(code, indent_first_line=True):
    if indent_first_line:
        code = '  ' + code
    return code.replace('\n', '\n  ')


device_function_template = """\
template <%(template_parameters)s>
struct %(function_name)s_functor : public thrust::unary_function<int, OutputType> {
    %(struct_members)s
    %(function_name)s_functor(%(static_variables)s) %(variable_initializations)s {}

    template <typename Tuple>
    __host__ __device__
    OutputType operator()(Tuple _t) %(function_body)s
};
"""
def create_device_function(function_node):
  type_, name, parameters, block = function_node
  template_types = ['OutputType']
  struct_vars = []
  parameter_vars = []
  underscore_vars = []

  for parameter in parameters:
      if parameter.type in data_types:
          template_types.append('Input%sType' % len(template_types))
          struct_vars.append('Array2d<%s> %s;' % (template_types[-1], parameter.name))
          parameter_vars.append('Array2d<%s> _%s' % (template_types[-1], parameter.name))
          underscore_vars.append('%(name)s(_%(name)s)' % { 'name' : parameter.name })
      else:
          struct_vars.append('%s %s;' % (type_map[parameter.type], parameter.name))
          parameter_vars.append('%s _%s' % (type_map[parameter.type], parameter.name))
          underscore_vars.append('%(name)s(_%(name)s)' % { 'name' : parameter.name})

  #if not (len(parameters) == 1 or parameters[0][0] == 'window'):
  #  raise Exception('Error, parameters %s must instead be window')


  if len(parameters) > 0:
      struct_vars = indent('\n'.join(struct_vars), indent_first_line=False)
      parameter_vars = ', '.join(parameter_vars)
      underscore_vars = ": " + ', '.join(underscore_vars)
  else:
      struct_vars = parameter_vars = underscore_vars = ''

  template_types = ', '.join(map(lambda type: 'typename %s' % type, template_types))

  environment = { 'function_name' : name,
                  'function_body' : indent(block.to_cpp(), indent_first_line=False),
                  'template_parameters' : template_types,
                  'struct_members' : struct_vars,
                  'static_variables' : parameter_vars,
                  'variable_initializations' : underscore_vars }

  return device_function_template % environment



type_map = {
        'Int' : 'int',
        'IntArray1d' : 'int',
        'IntArray2d' : 'int',
        'IntArray3d' : 'int',
        'Real' : 'float',
        'RealArray1d' : 'float',
        'RealArray2d' : 'float',
        'RealArray3d' : 'float',
        'Color' : 'uchar4',
        'ColorArray1d' : 'uchar4',
        'ColorArray2d' : 'uchar4',
        'ColorArray3d' : 'uchar4',
        'Bool' : 'bool',
        'BoolArray1d' : 'bool',
        'BoolArray2d' : 'bool',
        'BoolArray3d' : 'bool' }

data_to_scalar = type_map


numpy_type_map = {
        'Int' : numpy.int32,
        'IntArray1d' : numpy.int32,
        'IntArray2d' : numpy.int32,
        'IntArray3d' : numpy.int32,
        'Real' : numpy.float32,
        'RealArray1d' : numpy.float32,
        'RealArray2d' : numpy.float32,
        'RealArray3d' : numpy.float32,
        'Color' : types.uchar4,
        'ColorArray1d' : types.uchar4,
        'ColorArray2d' : types.uchar4,
        'ColorArray3d' : types.uchar4 }
        #'Bool' : vec.bool,
        #'Bool1d' : vec.bool,
        #'Bool2d' : vec.bool,
        #'Bool3d' : vec.bool }


dtype_to_ctype = {
        numpy.dtype('int32') : 'int',
        numpy.float32 : 'float',
        types.uchar4 : 'uchar4' }


data_create_template = """\
Array2d<%(type)s> %(name)s = _allocator.arrayWithSize<%(type)s>(%(width)s, %(height)s);
"""
def create_data(type_, name, size):
  return data_create_template % { 'type' : type_map[type_],
                                  'name' : name,
                                  'width' : size.width,
                                  'height' : size.height }


data_release_template = """\
_allocator.releaseArray(%(name)s);
"""
def release_data(type_, name):
    return data_release_template % { 'type' : type_map[type_],
                                     'name' : name }

data_swap_template = """\
{first_data}.swapDataWith({second_data});
"""
def swap_data(first_data, second_data):
    return data_swap_template.format(first_data=first_data, second_data=second_data)

