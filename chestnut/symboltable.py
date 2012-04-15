from collections import namedtuple
import numpy
from pycuda import gpuarray

def combinations(ones, twos):
    for a in ones:
        for b in twos:
            yield a, b

def array_type(scalar_type, dimension):
    return '{0}Array{1}d'.format(scalar_type, dimension)

def array_element_type(scalar_type, dimension):
    return '{0}{1}d'.format(scalar_type, dimension)

scalar_types = ['Int', 'Real', 'Color', 'Bool']

dimensions = ['1', '2', '3']

# IntArray1d, ColorArray2d, etc...
array_types = [array_type(t, d) for t, d in combinations(scalar_types, dimensions)]

# Int1d, Color2d, etc...
array_element_types = [array_element_type(t, d) for t, d in combinations(scalar_types, dimensions)]

reserved_words = [
            'parallel',
            'sequential',

            'if',
            'else',
            'while',
            'for',
            'foreach',
            'end',
            'break',
            'return',
            ]

keywords = scalar_types + array_types + array_element_types + reserved_words


numpy_type_map = { 'int2d' : numpy.int32,
                   'real2d' : numpy.float32 }

class UninitializedException(Exception): pass

DisplayWindow = namedtuple('DisplayWindow', ['name', 'title', 'width', 'height'])

Keyword = namedtuple('Keyword', ['name'])
SequentialFunction = namedtuple('SequentialFunction', ['name', 'type', 'parameters', 'node'])
ParallelFunction = namedtuple('ParallelFunction', ['name', 'type', 'parameters', 'node'])

Object = namedtuple('Object', ['name'])

class NeutralFunction(namedtuple('NeutralFunction', ['name', 'type', 'parameters', 'node'])):
    @property
    def cpp_name(self):
        return self.name

class ArrayElement(namedtuple('ArrayElement', ['name', 'type', 'array', 'cpp_name'])): pass


class Variable(namedtuple('Variable', ['name', 'type'])):
    @property
    def cpp_name(self):
        return self.name

    @property
    def value(self):
        try:
            return self._value
        except AttributeError:
            raise UninitializedException('Variable %s was accessed before it was initialized' % self.name)
    @value.setter
    def value(self, new_value):
        self._value = new_value

class Window(namedtuple('Window', ['name', 'number'])):
    @property
    def value(self):
        try:
            return self._value
        except AttributeError:
            raise UninitializedException('Window %s was accessed before it was initialized' % self.name)
    @value.setter
    def value(self, new_value):
        self._value = new_value

class Array(namedtuple('Array', ['name', 'type', 'size'])):
    @property
    def cpp_name(self):
        return self.name

    @property
    def has_display(self):
        try:
            return self._has_display
        except AttributeError:
            self._has_display = 0
            return self._has_display
    @has_display.setter
    def has_display(self, value):
        self._has_display = value

    def __create(self):
        self._array = numpy.zeros((self.height, self.width), dtype=numpy_type_map[self.type])
    @property
    def length(self):
        return self.size.length
    @property
    def array(self):
        return self.value
    @property
    def value(self):
        try:
            return self._array
        except AttributeError:
            self.__create()
            return self._array
    @value.setter
    def value(self, new_data):
        self._array = new_data

    def at(self, x, y):
        try:
            return self._array[y, x]
        except AttributeError:
            self.__create()
            return self._array[y, x]
    def setAt(self, x, y, value):
        try:
            self._array[y, x] = value
        except AttributeError:
            self.__create()
            self._array[y, x] = value

class EntryExistsError(Exception):
  def __init__(self, name):
    self.name = name
  def __str__(self):
    return "Error, %s already exists in this scope" % self.name

class CompileTimeScope(dict):
    def add(self, entry):
        if entry.name in self:
            raise EntryExistsError(entry.name)
        self[entry.name] = entry
        return entry

    def insert(self, name, entry):
        if name in self:
            raise EntryExistsError(name)
        self[name] = entry
        return entry

    def remove(self, name):
        if name in self:
            del self[name]


class CompileTimeParallelScope(CompileTimeScope):
    pass

class Scope(dict):
    def __init__(self, parent=None):
        self.parent = parent

    def __setitem__(self, name, value):
        if name in self:
            raise EntryExistsError(entry.name)
        dict.__setitem__(self, name, value)
        return value

    def __getitem__(self, name):
        if name in self:
            return dict.__getitem__(self, name)
        else:
            if not self.parent:
                raise Exception("Variable %s not found in the current scope" % name) # TODO: Turn into Chestnut exception

            return self.parent[name]

class SymbolTable(object):
    def __init__(self):
        self.unique_counter = 0
        self.table = []
        self.structures = []
        self.displayWindows = []
        self.parallelContexts = []
        self.parallelFunctionDeclarations = []
        self.sequentialFunctionDeclarations = []
        self.createScope() # global scope

        for word in keywords:
            self.currentScope.add(Keyword(word))

    def __str__(self):
        result = "\nSymbol Table\n"
        for entry in self.table:
            result += str(entry) + "\n"
        return result

    def uniqueNumber(self):
        self.unique_counter += 1
        return str(self.unique_counter)

    def add(self, entry):
        return self.currentScope.add(entry)

    def insert(self, name, entry):
        return self.currentScope.insert(name, entry)

    def remove(self, name):
        return self.currentScope.remove(name)

    @property
    def in_parallel_context(self):
        try:
            return isinstance(self.currentScope, CompileTimeParallelScope)
        except IndexError: # no scopes yet
            return False

    def createScope(self):
        if self.in_parallel_context:
            return self.createParallelScope()

        self.table.insert(0, CompileTimeScope())

    def createParallelScope(self):
        self.table.insert(0, CompileTimeParallelScope())

    def removeScope(self):
        self.table.pop(0)

    @property
    def currentScope(self):
        return self.table[0]

    def lookup(self, name):
        for scope in self.table:
            try:
                if name in scope:
                    return scope[name]
            except TypeError: # unhashable type
                return None
        return None

class SymbolTableError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

def main():
    table = SymbolTable()
    print table

if __name__ == '__main__':
    main()

