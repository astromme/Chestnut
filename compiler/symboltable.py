from collections import namedtuple
import numpy

# Keywords supported by chestnut syntax
# These are added to the symbol table
scalar_types = ['Integer', 'Real', 'Color', 'Bool']
data_types = ['Integer1d', 'Integer2d', 'Integer3d',
              'Real1d', 'Real2d', 'Real3d',
              'Color1d', 'Color2d', 'Color3d',
              'Bool1d', 'Bool2d', 'Bool3d']
structure_types = ['Size1', 'Size2', 'Size3',
                   'Window']

reserved_words = [
            'parallel',
            'sequential',

            'if',
            'else',
            'while',
            'for',
            'break',
            'return',
            ]

keywords = scalar_types + data_types + structure_types + reserved_words


numpy_type_map = { 'int2d' : numpy.int32,
                   'real2d' : numpy.float32 }

class UninitializedException(Exception): pass

DisplayWindow = namedtuple('DisplayWindow', ['name', 'title', 'width', 'height'])

Keyword = namedtuple('Keyword', ['name'])
SequentialFunction = namedtuple('SequentialFunction', ['name', 'type', 'parameters', 'ok_for_device', 'node'])
ParallelFunction = namedtuple('ParallelFunction', ['name', 'type', 'parameters', 'node'])

class Variable(namedtuple('Variable', ['name', 'type'])):
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

class Data(namedtuple('Data', ['name', 'type', 'width', 'height'])):
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
    def size(self):
        return self.length
    @property
    def length(self):
        return self.width * self.height
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

class Scope(dict):
  def add(self, entry):
    if entry.name in self:
        raise EntryExistsError(entry.name)
    self[entry.name] = entry
    return entry

class SymbolTable:
    def __init__(self):
        self.table = []
        self.displayWindows = []
        self.createScope() # global scope

        for word in keywords:
            self.currentScope.add(Keyword(word))

    def __str__(self):
        result = "\nSymbol Table\n"
        for entry in self.table:
            result += str(entry) + "\n"
        return result

    def add(self, entry):
        return self.currentScope.add(entry)

    def createScope(self):
      self.table.insert(0, Scope())

    def removeScope(self):
        self.table.pop(0)

    @property
    def currentScope(self):
        return self.table[0]

    def lookup(self, name):
        for scope in self.table:
            if name in scope:
                return scope[name]
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

