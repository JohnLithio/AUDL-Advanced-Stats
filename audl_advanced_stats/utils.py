import sqlite3
from os.path import basename, join
from sqlite3 import Error


def get_database_path(database_path="data"):
    """Return the file path to the SQLite database."""
    return database_path


def get_json_path(database_path, folder):
    """Return the file path to the JSON game data."""
    return join(database_path, folder)


def create_connection(database_path, database_name="audl.db"):
    """Create a connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(join(database_path, database_name))
        return conn
    except Error as e:
        print(e)

    return conn
