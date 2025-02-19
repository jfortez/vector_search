import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
import logging

from models.identificacion import Identificacion, IdentificacionCreate
from database.connection import Database


class IdentificacionDAO:
    """Manejo de la tabla Personas."""

    def __init__(self, db: Database):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def find_by_id(self, id: int) -> Identificacion:
        """Obtiene una identificación por su ID."""
        try:
            with self.db.get_connection() as conn:
                query = text("SELECT * FROM Personas WHERE id = :id")
                result = conn.execute(query, {"id": id})
                return Identificacion(**result.fetchone())
        except SQLAlchemyError as e:
            self.logger.error(f"Error al obtener identificación: {str(e)}")
            raise

    def get_all(self) -> pd.DataFrame:
        """Obtiene todas las identificaciones."""
        try:
            with self.db.get_connection() as conn:
                query = text("SELECT id, identificacion, nombre FROM Personas")
                return pd.read_sql(query, conn)
        except SQLAlchemyError as e:
            self.logger.error(f"Error al obtener identificaciones: {str(e)}")
            raise

    def insert(self, new_row: IdentificacionCreate) -> int:
        """Inserta una nueva identificación."""
        try:
            with self.db.get_connection() as conn:
                query = text(
                    """
                    SET NOCOUNT ON;
                    INSERT INTO Personas (identificacion, nombre)
                    OUTPUT INSERTED.id
                    VALUES (:identificacion, :nombre)
                    """
                )
                result = conn.execute(
                    query,
                    {
                        "identificacion": new_row.identificacion,
                        "nombre": new_row.nombre,
                    },
                )
                inserted_id = result.fetchone()
                conn.commit()
                return inserted_id[0] if inserted_id else None
        except SQLAlchemyError as e:
            self.logger.error(f"Error al insertar identificación: {str(e)}")
            raise

    def delete(self, id: int) -> bool:
        """
        Elimina un registro de identificación por ID.

        Args:
            id: ID del registro a eliminar.

        Returns:
            bool: True si se eliminó exitosamente, False en caso contrario.

        Raises:
            SQLAlchemyError: Si ocurre un error en la base de datos.
        """
        try:
            with self.db.get_connection() as conn:
                query = text("DELETE FROM Personas WHERE id = :id")
                result = conn.execute(query, {"id": id})
                conn.commit()
                return result.rowcount > 0
        except SQLAlchemyError as e:
            self.logger.error(f"Error al eliminar identificación {id}: {str(e)}")
            raise

    def update(self, row: Identificacion) -> bool:
        """
        Actualiza un registro de identificación.

        Args:
            row: Objeto Identificacion con los datos a actualizar.

        Returns:
            bool: True si se actualizó exitosamente, False en caso contrario.

        Raises:
            SQLAlchemyError: Si ocurre un error en la base de datos.
        """
        try:
            with self.db.get_connection() as conn:
                query = text(
                    """
                    UPDATE Personas 
                    SET identificacion = :identificacion, 
                        nombre = :nombre 
                    WHERE id = :id
                """
                )
                result = conn.execute(
                    query,
                    {
                        "id": row.id,
                        "identificacion": row.identificacion,
                        "nombre": row.nombre,
                    },
                )
                conn.commit()
                return result.rowcount > 0
        except SQLAlchemyError as e:
            self.logger.error(f"Error al actualizar identificación {row.id}: {str(e)}")
            raise
