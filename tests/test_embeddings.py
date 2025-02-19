from embedding import FaissIndexManager  # Asegúrate de importar la clase
import pandas as pd

# Crear una instancia del administrador de índices
manager = FaissIndexManager()


def test_insert_data():
    new_data = pd.DataFrame(
        [{"id": 999, "identificacion": "99999999", "nombre": "Test User"}]
    )
    initial_length = len(manager.data)
    manager.insert_data(new_data)
    assert len(manager.data) == initial_length + len(new_data)


def test_delete_data():
    initial_length = len(manager.data)
    ids_to_delete = manager.data["id"].tolist()[:1]  # Eliminar el primer ID
    manager.delete_data(ids_to_delete)
    assert len(manager.data) == initial_length - 1


def test_update_data():
    updated_data = pd.DataFrame(
        [{"id": 999, "identificacion": "updated_id", "nombre": "Updated User"}]
    )
    manager.insert_data(updated_data)  # Asegúrate de que el ID 999 existe
    manager.update_data(updated_data)
    assert (
        manager.data.loc[manager.data["id"] == 999, "nombre"].values[0]
        == "Updated User"
    )
