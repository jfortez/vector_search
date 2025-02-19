import pytest
from database.connection import Database
from models.identificacion import Identificacion
from database.dao.identificacion import IdentificacionDAO
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
import pandas as pd
from util import generate_unique_id


@pytest.fixture(scope="function")
def db():
    """Fixture que proporciona una instancia de Database y limpia después de cada prueba."""
    database = Database()
    identificacion = IdentificacionDAO(database)
    yield identificacion
    # Limpiamos todos los datos de prueba
    with database.get_connection() as conn:
        conn.execute(
            text(
                """
                DELETE FROM Personas 
                WHERE nombre LIKE 'Test User%' 
                OR nombre LIKE 'Test_%'
                """
            )
        )
        conn.commit()


def test_get_identificacion(db):
    """Prueba la obtención de registros."""
    df = db.get_all()
    assert isinstance(df, pd.DataFrame), "Debería retornar un DataFrame"
    assert "id" in df.columns, "Debería tener una columna 'id'"
    assert "identificacion" in df.columns, "Debería tener una columna 'identificacion'"
    assert "nombre" in df.columns, "Debería tener una columna 'nombre'"


def test_insert_identificacion(db):
    """Prueba la inserción de un nuevo registro."""
    unique_id = generate_unique_id()

    # Insertamos un nuevo registro
    new_id = db.insert(unique_id, "Test User")

    # Verificamos que se haya insertado correctamente
    assert new_id is not None, "Debería retornar el ID del nuevo registro"

    # Verificamos que podemos recuperar el registro
    df = db.get_all()
    new_record = df[df["id"] == new_id]
    assert not new_record.empty, "El registro debería existir en la base de datos"
    assert new_record.iloc[0]["identificacion"] == unique_id
    assert new_record.iloc[0]["nombre"] == "Test User"


def test_insert_duplicate_identificacion(db):
    """Prueba que no se pueden insertar identificaciones duplicadas."""
    unique_id = generate_unique_id()

    # Primera inserción
    db.insert(unique_id, "Test User 1")

    # Segunda inserción con la misma identificación
    with pytest.raises(IntegrityError):
        db.insert(unique_id, "Test User 2")


def test_update_identificacion(db):
    """Prueba la actualización de un registro."""
    # Primero insertamos un registro
    unique_id = generate_unique_id()
    new_id = db.insert(unique_id, "Test User")

    # Actualizamos el registro
    updated_nombre = "Test Updated User"
    updated_identificacion = generate_unique_id()

    row = Identificacion(
        id=new_id, identificacion=updated_identificacion, nombre=updated_nombre
    )

    success = db.update(row)
    assert success, "La actualización debería ser exitosa"

    # Verificamos la actualización
    df = db.get_all()
    updated_record = df[df["id"] == new_id].iloc[0]
    assert updated_record["nombre"] == updated_nombre
    assert updated_record["identificacion"] == updated_identificacion


def test_delete_identificacion(db):
    """Prueba la eliminación de un registro."""
    # Primero insertamos un registro
    unique_id = generate_unique_id()
    new_id = db.insert(unique_id, "Test User")

    # Verificamos que existe
    df_before = db.get_all()
    assert len(df_before[df_before["id"] == new_id]) == 1

    # Eliminamos el registro
    success = db.delete(new_id)
    assert success, "La eliminación debería ser exitosa"

    # Verificamos que ya no existe
    df_after = db.get_all()
    assert len(df_after[df_after["id"] == new_id]) == 0


def test_delete_nonexistent_identificacion(db):
    """Prueba intentar eliminar un registro que no existe."""
    success = db.delete(99999)
    assert not success, "Debería retornar False al intentar eliminar un ID inexistente"
