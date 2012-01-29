from nodes import NeutralFunction, Parameter

#SequentialFunction = namedtuple('SequentialFunction', ['name', 'type', 'parameters', 'ok_for_device', 'node'])

built_in_functions = [NeutralFunction('sqrt',  'float', [Parameter(['float', 'value'])], None),
                      NeutralFunction('sqrtf', 'float', [Parameter(['float', 'value'])], None),
                      NeutralFunction('cos',   'float', [Parameter(['float', 'value'])], None),
                      NeutralFunction('sin',   'float', [Parameter(['float', 'value'])], None),
                      NeutralFunction('min',   'float', [Parameter(['float', 'value']), Parameter(['float', 'value'])], None),
                      NeutralFunction('max',   'float', [Parameter(['float', 'value']), Parameter(['float', 'value'])], None),
                      NeutralFunction('abs',   'float', [Parameter(['float', 'value'])], None),
                      NeutralFunction('exp',   'float', [Parameter(['float', 'value'])], None)]
