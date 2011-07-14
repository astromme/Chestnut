from symboltable import scalar_types, data_types
import numpy
from pycuda.gpuarray import vec as types

#helper to do nice code indenting.. duplicated in nodes, yuck
def indent(code, indent_first_line=True):
    if indent_first_line:
        code = '  ' + code
    return code.replace('\n', '\n  ')


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
        'BoolArray3d' : 'bool',
        'Size1d' : 'Size1d',
        'Size2d' : 'Size2d',
        'Size3d' : 'Size3d',
        'Point1d' : 'Point1d',
        'Point2d' : 'Point2d',
        'Point3d' : 'Point3d',
        'Point4d' : 'Point4d',
        'IntWindow2d' : 'Window2d<int>',
        'RealWindow2d' : 'Window2d<float>',
        'ColorWindow2d' : 'Window2d<color>',
        }

data_to_scalar = type_map
chestnut_to_c = type_map

datatype_to_windowtype = {
        'IntArray2d' : 'IntWindow2d',
        'RealArray2d' : 'RealWindow2d',
        'ColorWindow2d' : 'ColorWindow2d',
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

