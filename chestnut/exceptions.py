#helper functions to check for errors and to print them in a nice way
class CompilerException(Exception):
    def __init__(self, message, node=None):
        self.message = message
        self.node = node

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if not self.node or not self.node.context:
            return "Error\n  {message}".format(message=self.message)
        elif self.node.context.start_line == self.node.context.end_line:
            return "Error around line {line_num}\n  {message}".format(line_num=self.node.context.start_line,
                                                                            message=self.message)
        else:
            return "Error around lines {start} and {end}\n  {message}".format(start=self.node.context.start_line,
                                                                            end=self.node.context.end_line,
                                                                            message=self.message)


class InternalException(Exception): pass

#helper functions to emulate return, break
class InterpreterException(Exception): pass
class InterpreterReturn(Exception): pass
class InterpreterBreak(Exception): pass
