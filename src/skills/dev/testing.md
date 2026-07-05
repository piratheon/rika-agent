# skill: testing
description: Write and run automated tests using pytest; generate test cases for existing code.
tools: run_python, run_shell_command, write_file

## When to use
- Writing unit or integration tests for generated code
- Running an existing test suite and reporting results
- Generating test cases from function signatures or docstrings

## Usage pattern
```python
# Run tests with coverage
run_shell_command("pytest tests/ -v --tb=short --cov=src --cov-report=term-missing")

# Run a single test file
run_shell_command("pytest tests/test_module.py -v")

# Run tests matching a keyword
run_shell_command("pytest -k 'test_parse or test_format' -v")

# Write a test file
write_file(path="tests/test_example.py", content='''
import pytest
from src.module import MyClass

def test_basic_case():
    obj = MyClass()
    result = obj.process("input")
    assert result == "expected"

def test_edge_case_empty():
    obj = MyClass()
    with pytest.raises(ValueError):
        obj.process("")
''')
```

## Notes
- Install: `pip install pytest pytest-cov pytest-asyncio`
- For async functions, decorate with `@pytest.mark.asyncio`
- `--tb=short` is sufficient for quick diagnosis; use `--tb=long` for deep tracebacks
