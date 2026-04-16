"""
SQL Factory for managing database connections and executing queries.
This factory uses SQLAlchemy to connect to PostgreSQL database and execute queries.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import os

class SQLFactory:
    def __init__(self):
        """
        Initialize the SQL Factory with database path from DATABASE_URL environment variable
        """
        # Use DATABASE_URL environment variable for PostgreSQL connection
        self.db_path = os.getenv('DATABASE_URL')
        
        # Check if DATABASE_URL is set and not empty
        if not self.db_path:
            raise ValueError("DATABASE_URL environment variable is not set. Please configure the database connection URL.")
        
        # Create engine and session factory
        self.engine = create_engine(self.db_path)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def execute_query(self, query):
        """
        Execute a SQL query and return results as JSON array
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            list: Array of dictionaries representing the query results
        """
        try:
            # Create a new session
            db_session = self.SessionLocal()
            
            # Execute the query
            result = db_session.execute(text(query))
            
            # Get column names
            columns = result.keys()
            
            # Convert to array of JSON objects
            rows = []
            for row in result:
                row_dict = dict(zip(columns, row))
                rows.append(row_dict)
            
            return rows
            
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")
        finally:
            db_session.close()
    
    def get_engine(self):
        """
        Get the SQLAlchemy engine
        
        Returns:
            Engine: The SQLAlchemy engine instance
        """
        return self.engine

# Create a singleton instance
sql_factory = SQLFactory()