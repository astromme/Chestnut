from symboltable import scalar_types, dimensions, array_type, array_types, array_element_type, array_element_types, combinations
import numpy
from pycuda.gpuarray import vec as types

#helper to do nice code indenting.. duplicated in nodes, yuck
def indent(code, indent_first_line=True):
    if indent_first_line:
        code = '  ' + code
    return code.replace('\n', '\n  ')

#  { 'IntArray1d' : 'Int', 'BoolArray3d' : 'Bool', etc... }
array_to_scalar = dict([(array_type(type, dimension), type)
                        for type, dimension in combinations(scalar_types, dimensions)])


#  { 'IntArray1d' : 'Int1d', 'BoolArray3d' : 'Bool3d', etc... }
array_to_array_element = dict([(array_type(type, dimension), array_element_type(type, dimension))
                               for type, dimension in combinations(scalar_types, dimensions)])

def c_type_from_chestnut_type(type_name):
    if type_name in array_types:
        return array_to_scalar[type_name]
    else:
        return type_name

# TODO: Update with automatic mapping
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

coordinates = {
        'x' : '_x',
        'y' : '_y',
        'z' : '_z',
        'width' : '_width',
        'height' : '_height',
        'depth' : '_depth' }

color_properties = { # Screen is apparently BGR not RGB
        'red' : 'z',
        'green' : 'y',
        'blue' : 'x',
        'opacity' : 'w' }


# TODO: Update with automatic mapping
dtype_to_ctype = {
        numpy.dtype('int32') : 'int',
        numpy.float32 : 'float',
        types.uchar4 : 'Color' }


def create_data_like_data(name, like_data):
    return data_create_template % { 'type' : data_to_scalar[like_data.type],
                                    'name' : name,
                                    'dimensions' : '{0.name}.size()'.format(like_data) }


data_release_template = """\
_allocator.releaseArray(%(name)s);
"""
def release_data(type_, name):
    return data_release_template % { 'type' : data_to_scalar[type_],
                                     'name' : name }

data_swap_template = """\
{first_data}.swapDataWith({second_data});
"""
def swap_data(first_data, second_data):
    return data_swap_template.format(first_data=first_data, second_data=second_data)

