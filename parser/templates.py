def pad(value):
  return value+2

device_function_template = """\
struct %(function_name)s_functor
{
    %(struct_members)s

    %(function_name)s_functor(%(static_variables)s) %(variable_initializations)s {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
      /*
      int index = thrust::get<2>(t);
      int paddedWidth = thrust::get<3>(t);
      int paddedHeight = thrust::get<4>(t);

      int x = index %% paddedWidth - 1; // -1 for padding
      int y = index / paddedWidth - 1;

      int width = paddedWidth - 2;
      int height = paddedHeight - 2;
      */

      %(function_body)s
    }
};
"""
def create_device_function(function_node):
  name, parameters, block = function_node
  static_variables = []
  static_types = []

  for parameter in parameters:
      if not parameter.type == 'window':
          static_variables.append(parameter.name)
          static_types.append(parameter.type)

  #if not (len(parameters) == 1 or parameters[0][0] == 'window'):
  #  raise Exception('Error, parameters %s must instead be window')

  underscore_vars = map(lambda var: var + "_", static_variables)

  struct_members = ';\n'.join(map(lambda tup: "%s %s" % tup, zip(static_types, static_variables)))
  variable_initializations = ', '.join(map(lambda tup: "%s(%s)" % tup, zip(static_variables, underscore_vars)))
  if len(static_variables) > 0:
      struct_members += ';'
      variable_initializations = ": " + variable_initializations


  environment = { 'function_name' : name,
                  'function_body' : block.to_cpp(),
                  'struct_members' : struct_members,
                  'static_variables' : ', '.join(map(lambda tup: "%s %s" % tup, zip(static_types, underscore_vars))),
                  'variable_initializations' : variable_initializations }

  return device_function_template % environment



type_map = { 'real' : 'float',
             'int' : 'int',
             'real2d' : 'float',
             'int2d' : 'int',
             'void' : 'void' }

data_template = """\
Chestnut<%(type)s>::DeviceData %(name)s(%(width)s, %(height)s);
"""
def create_data(type_, name, size):
  return data_template % { 'type' : type_map[type_],
                           'name' : name,
                           'width' : pad(size.width),
                           'height' : pad(size.height) }

