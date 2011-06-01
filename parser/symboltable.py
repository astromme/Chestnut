from collections import namedtuple

# Keywords supported by chestnut syntax
# These are added to the symbol table
keywords = ['int',
            'char',
            'if',
            'else',
            'while',
            'for',
            'return',
            'break',
            ]


Keyword = namedtuple('Keyword', ['name'])
Variable = namedtuple('Variable', ['name', 'type'])
Data = namedtuple('Data', ['name', 'type', 'width', 'height'])
Array = namedtuple('Array', ['name', 'type'])
SequentialFunction = namedtuple('SequentialFunction', ['name', 'type', 'parameters', 'ok_for_device'])
ParallelFunction = namedtuple('ParallelFunction', ['name', 'parameters'])

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

