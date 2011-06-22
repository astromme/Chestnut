import pytest
from chestnut.parser import parse

def test_unclosed_size_definition_at_semicolon():
    with pytest.raises(SyntaxError) as error:
        parse('Integer2d data[5, 10;')
    assert error.value.lineno == 1
    assert str(error.value) == "Datablock size specification is missing a closing ] (<string>, line 1)"

def test_unclosed_block_definition_at_eof():
    with pytest.raises(SyntaxError) as error:
        parse('{ int x;')
    assert error.value.lineno == 1
    assert str(error.value) == 'block is missing a closing } (<string>, line 1)'


if __name__ == '__main__':
    test_unclosed_block_definition_at_eof()
