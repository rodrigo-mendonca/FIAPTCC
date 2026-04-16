# SQL Factory Tests

This directory contains tests for the SQL factory that manages database connections and executes queries.

## Test Files

1. **test_sql_factory.py** - Basic unit tests for the SQL factory functionality
2. **test_sql_factory_integration.py** - Integration tests that verify proper query execution 
3. **test_database_structure.py** - Tests that verify database structure compatibility

## Running Tests

To run all tests:
```bash
cd tests/sql
python -m pytest -v
```

Or use the test runner script:
```bash
cd tests/sql
python run_tests.py
```

## Test Coverage

The tests cover:

- SQL factory initialization with DATABASE_URL environment variable
- Query execution functionality 
- Error handling for database connection failures
- Database structure verification
- Mocked database interactions to ensure proper method behavior

## Dependencies

Tests require:
- pytest
- mock
- sqlalchemy (from main project)