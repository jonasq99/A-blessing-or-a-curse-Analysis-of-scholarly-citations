current tests are marked with a @pytest.mark.correct decorator. This is because pytest
was sometimes running tets from the broken_tests folder in parallel. To run tests isolated,
the decorator has been added to tests of interest during development.
To run tests with decorators use:
pytest -m correct tests/test_whichfileever.py