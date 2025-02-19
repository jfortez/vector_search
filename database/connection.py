from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from contextlib import contextmanager
import logging
from typing import Optional
from .config import DatabaseConfig


class Database:
    """Clase para manejar la conexión a la base de datos."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine = self._create_engine()
        self.logger = logging.getLogger(__name__)

    def _create_engine(self) -> Engine:
        """Crea y retorna el engine de SQLAlchemy."""
        connection_string = self.config.get_connection_string()
        return create_engine(
            f"mssql+pyodbc:///?odbc_connect={connection_string}",
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    @contextmanager
    def get_connection(self):
        """Maneja conexiones de base de datos."""
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()

    def __del__(self):
        """Cierra el engine al destruir la instancia."""
        if hasattr(self, "engine"):
            try:
                self.engine.dispose()
            except Exception as e:
                # Log the exception if needed
                logging.error(
                    f"Error al cerrar la conexión de la base de datos: {str(e)}"
                )
