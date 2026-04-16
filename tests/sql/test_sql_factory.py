import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fiap_api.factories.sql_factory import SQLFactory

class TestSQLFactory:
    """Test cases for SQL Factory"""
    
    def test_sql_factory_initialization(self):
        """Test that SQL factory initializes correctly with DATABASE_URL"""
        # Mock the environment variable
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/db'}):
            sql_factory = SQLFactory()
            
            assert sql_factory.db_path == 'postgresql://user:pass@localhost/db'
            assert hasattr(sql_factory, 'engine')
            assert hasattr(sql_factory, 'SessionLocal')
    
    def test_sql_factory_initialization_missing_env_var(self):
        """Test that SQL factory raises error when DATABASE_URL is not set"""
        # Mock the environment variable to be empty
        with patch.dict(os.environ, {'DATABASE_URL': ''}):
            with pytest.raises(ValueError) as excinfo:
                sql_factory = SQLFactory()
            
            assert "DATABASE_URL environment variable is not set" in str(excinfo.value)
    
    def test_execute_query_success(self):
        """Test successful query execution"""
        # Mock the database connection and session
        mock_engine = MagicMock()
        mock_session = MagicMock()
        
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/db'}):
            sql_factory = SQLFactory()
            
            # Mock engine and session creation
            with patch.object(sql_factory, 'engine', mock_engine):
                with patch.object(sql_factory.SessionLocal, 'return_value', mock_session):
                    # Mock the result of query execution
                    mock_result = MagicMock()
                    mock_result.keys.return_value = ['id', 'name']
                    mock_result.__iter__.return_value = [[1, 'Test']]
                    mock_session.execute.return_value = mock_result
                    
                    # Execute a test query
                    result = sql_factory.execute_query("SELECT * FROM usuarios")
                    
                    assert isinstance(result, list)
                    assert len(result) == 1
                    assert result[0]['id'] == 1
                    assert result[0]['name'] == 'Test'
    
    def test_execute_query_exception_handling(self):
        """Test that SQL factory properly handles query execution errors"""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/db'}):
            sql_factory = SQLFactory()
            
            # Mock engine and session to raise an exception
            mock_engine = MagicMock()
            mock_session = MagicMock()
            
            with patch.object(sql_factory, 'engine', mock_engine):
                with patch.object(sql_factory.SessionLocal, 'return_value', mock_session):
                    mock_session.execute.side_effect = Exception("Database connection failed")
                    
                    # Test that the exception is properly raised
                    with pytest.raises(Exception) as excinfo:
                        sql_factory.execute_query("SELECT * FROM usuarios")
                    
                    assert "Error executing query" in str(excinfo.value)
    
    def test_get_engine(self):
        """Test that get_engine method returns the engine"""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/db'}):
            sql_factory = SQLFactory()
            
            engine = sql_factory.get_engine()
            
            assert engine == sql_factory.engine