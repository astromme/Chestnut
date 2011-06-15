from nodes import SequentialFunction, Parameter

#SequentialFunction = namedtuple('SequentialFunction', ['name', 'type', 'parameters', 'ok_for_device', 'node'])

built_in_functions = [SequentialFunction('sqrt',  'float', [Parameter(['float', 'value'])], True, None),
                      SequentialFunction('sqrtf', 'float', [Parameter(['float', 'value'])], True, None),
                      SequentialFunction('cos',   'float', [Parameter(['float', 'value'])], True, None),
                      SequentialFunction('sin',   'float', [Parameter(['float', 'value'])], True, None)]
