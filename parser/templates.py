def pad(value):
  return value+2

device_function_template = """
struct %(function_name)s_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
      %(function_body)s
    }
};
"""
def create_device_function(function_node):
  name, parameters, block = function_node
  if not (len(parameters) == 1 or parameters[0][0] == 'window'):
    raise Exception('Error, parameters %s must instead be window')

  environment = { 'function_name' : name,
                  'function_body' : block.to_cpp() }

  return device_function_template % environment


type_map = { 'real2d' : 'float',
             'int2d' : 'int' }
data_template = """
DeviceData<%(type)s> %(name)s(%(width)s, %(height)s);
"""
def create_data(type_, name, size):
  return data_template % { 'type' : type_map[type_],
                           'name' : name,
                           'width' : pad(size.width),
                           'height' : pad(size.height) }

