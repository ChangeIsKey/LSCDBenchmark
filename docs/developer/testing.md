# Testing

There are two main kind of testings: unit testing and integration testing. We use both to test the LSCDBenchmark. There will be a short explaination and a tutorial with example script for how to do the testing.

```{warning}
Make sure you run all the tests in the main directory. 
```

## Unit Test

In unit test, we only test one small unit in a function or methed. For example, you want to test 3 string methods: string.upper(), string.isupper(), string.split(). You can have following example script in `test.py`.

```python
import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
```

There are 3 assert methods, provided by the TestCase class, in the script. They are used to check for and report failures. You can see the [list of methods](https://docs.python.org/3/library/unittest.html#classes-and-functions) for more assert methods.

```{seealso}
See [the documentation page](https://docs.python.org/3/library/unittest.html) about unit test in Python for more detail in implementation.
```

````{note}
We have the source code inside the `src` directory and the test code in the `test` directory.

```markdown
LSCDBenchmark/
├── src/
│   └── module.py
└── tests/
    ├── integration/
    └── unit/
        ├── test_method_1.py
        └── test_method_2.py
```

An error can happen for import a module in `src` for testing. Please add the following two lines to make the system able to locate back to the main directory:

```python
import sys
sys.path.insert(0, ".")
```
````

## Integration Test

After building several unit-tests in one module, we set an integration test for testing the whole module or the cooperation between modules. Sometimes modules can work on their own, but there is no guarantee that they can be assembled to work at the same time.

To have a clear concept for testing benchmark, you can use `tests/integration/test_template.py` in our LSCDBenchmark repository to start writing the testing script.
