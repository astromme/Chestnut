import operator as op


class RuntimeWindow:
    def __init__(self, x, y, data):
        self.data = data
        self.x = x
        self.y = y

    def prop(self, prop):
        #TODO: Update for 3d
        properties = { 'topLeft' : self.topLeft,
                    'top' : self.top,
                    'topRight' : self.topRight,
                    'left' : self.left,
                    'center' : self.center,
                    'bottomLeft' : self.bottomLeft,
                    'bottom' : self.bottom,
                    'bottomRight' : self.bottomRight,
                    'x' : self.x,
                    'y' : self.y,
                    'width' : self.width,
                    'height' : self.height }

        return properties[prop]



    @property
    def width(self):
        return self.data.width
    @property
    def height(self):
        return self.data.height

    @property
    def topLeft(self):
        return self.data.at((self.x-1) % self.width, (self.y-1) % self.height)

    @property
    def top(self):
        return self.data.at((self.x) % self.width, (self.y-1) % self.height)

    @property
    def topRight(self):
        return self.data.at((self.x+1) % self.width, (self.y-1) % self.height)

    @property
    def left(self):
        return self.data.at((self.x-1) % self.width, (self.y) % self.height)

    @property
    def center(self):
        return self.data.at((self.x) % self.width, (self.y) % self.height)

    @property
    def right(self):
        return self.data.at((self.x+1) % self.width, (self.y) % self.height)

    @property
    def bottomLeft(self):
        return self.data.at((self.x-1) % self.width, (self.y+1) % self.height)

    @property
    def bottom(self):
        return self.data.at((self.x) % self.width, (self.y+1) % self.height)

    @property
    def bottomRight(self):
        return self.data.at((self.x+1) % self.width, (self.y+1) % self.height)

# Other structures
class Program(List):
    def evaluate(self, env):
        import pycuda.tools

        # effing stupid pycuda context crashes with abort() on deletion
        # so we can't use the nice with blah() as blah: syntax
        #with cuda_context() as context:
        #env['@context'] = context
        device_pool = env['@device_memory_pool'] = pycuda.tools.DeviceMemoryPool()
        host_pool = env['@host_memory_pool'] = pycuda.tools.PageLockedMemoryPool()

        for node in self:
            node.evaluate(env)


### Nodes ###
# Operators
class Not(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "!%s" % cpp_tuple(self, env)
    def evaluate(self, env):
        return not self[0].evaluate(env)
class Neg(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "-%s" % cpp_tuple(self, env)
    def evaluate(self, env):
        return -self[0].evaluate(env)


### Products ###
class Product(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s)" % '*'.join(cpp_tuple(self, env))
    def evaluate(self, env):
        return reduce(op.mul, map(lambda obj: obj.evaluate(env), self))
class Inverse(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(1.0/%s)" % self[0].to_cpp(env)
    def evaluate(self, env):
        return op.truediv(1, self[0].evaluate(env))


### Sums ###
class Sum(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s)" % '+'.join(cpp_tuple(self, env))
    def evaluate(self, env):
        return reduce(op.add, map(lambda obj: obj.evaluate(env), self))

class Negative(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(-%s)" % self[0].to_cpp(env)
    def evaluate(self, env):
        return op.neg(self[0].evaluate(env))

### Other Operators ###
class Mod(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "((int)%s %% %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) % self[1].evaluate(env)


### Comparetors ###
class LessThan(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s < %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) < self[1].evaluate(env)
class LessThanOrEqual(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s <= %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) <= self[1].evaluate(env)
class GreaterThan(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s > %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) > self[1].evaluate(env)
class GreaterThanOrEqual(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s >= %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) >= self[1].evaluate(env)

class Equal(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s == %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) == self[1].evaluate(env)
class NotEqual(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s != %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return not (self[0].evaluate(env) == self[1].evaluate(env))

class BooleanAnd(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s && %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) and self[1].evaluate(env)

class BooleanOr(List):
    def to_cpp(self, env=defaultdict(bool)):
        return "(%s || %s)" % cpp_tuple(self, env)
    def evaluate(self, env):
        return self[0].evaluate(env) or self[1].evaluate(env)

class Assignment(List):
    def to_cpp(self, env=defaultdict(bool)):
        output, expression = self
        env = defaultdict(bool, env, data_to_assign=output)
        return '%s = %s' % (output.to_cpp(env), expression.to_cpp(env))
    def evaluate(self, env):
        symbol = env[self[0]]
        symbol.value = self[1].evaluate(env)
        return symbol.value

class AssignPlus(List):
    def to_cpp(self, env=defaultdict(bool)):
        output, expression = self
        env = defaultdict(bool, env, data_to_assign=output)
        return '%s += %s' % (output.to_cpp(env), expression.to_cpp(env))
class AssignMinus(List):
    def to_cpp(self, env=defaultdict(bool)):
        output, expression = self
        env = defaultdict(bool, env, data_to_assign=output)
        return '%s -= %s' % (output.to_cpp(env), expression.to_cpp(env))
class AssignMul(List):
    def to_cpp(self, env=defaultdict(bool)):
        output, expression = self
        env = defaultdict(bool, env, data_to_assign=output)
        return '%s *= %s' % (output.to_cpp(env), expression.to_cpp(env))
class AssignDiv(List):
    def to_cpp(self, env=defaultdict(bool)):
        output, expression = self
        env = defaultdict(bool, env, data_to_assign=output)
        return '%s /= 1.0*%s' % (output.to_cpp(env), expression.to_cpp(env))
# End Operators
