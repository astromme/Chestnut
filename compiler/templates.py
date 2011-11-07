from symboltable import scalar_types, data_types
import numpy
from pycuda.gpuarray import vec as types

#helper to do nice code indenting.. duplicated in nodes, yuck
def indent(code, indent_first_line=True):
    if indent_first_line:
        code = '  ' + code
    return code.replace('\n', '\n  ')

data_to_scalar = {
        'IntArray1d' : 'Int',
        'IntArray2d' : 'Int',
        'IntArray3d' : 'Int',
        'RealArray1d' : 'Real',
        'RealArray2d' : 'Real',
        'RealArray3d' : 'Real',
        'ColorArray1d' : 'Color',
        'ColorArray2d' : 'Color',
        'ColorArray3d' : 'Color',
        'BoolArray1d' : 'Bool',
        'BoolArray2d' : 'Bool',
        'BoolArray3d' : 'Bool',
        }

datatype_to_windowtype = {
        'IntArray1d'    : 'IntWindow1d',
        'RealArray1d'   : 'RealWindow1d',
        'ColorArray1d'  : 'ColorWindow1d',
        'BoolArray1d'   : 'BoolWindow1d',

        'IntArray2d'    : 'IntWindow2d',
        'RealArray2d'   : 'RealWindow2d',
        'ColorArray2d'  : 'ColorWindow2d',
        'BoolArray2d'   : 'BoolWindow2d',

        'IntArray3d'    : 'IntWindow3d',
        'RealArray3d'   : 'RealWindow3d',
        'ColorArray3d'  : 'ColorWindow3d',
        'BoolArray3d'   : 'BoolWindow3d',
        }


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
        types.uchar4 : 'Color' }


data_create_template = """\
Array<%(type)s> %(name)s = _allocator.arrayWithSize<%(type)s>(%(dimensions)s);
"""
def create_data(type_, name, size):
    dimensions = size[0:size.dimension]
    return data_create_template % { 'type' : data_to_scalar[type_],
                                    'name' : name,
                                    'dimensions' : ', '.join(map(str, dimensions)) }

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

