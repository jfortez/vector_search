from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

from embedding import FaissIndexManager

router = APIRouter(prefix="/identificaciones", tags=["identificaciones"])

# Modelo Pydantic para los registros

manager = FaissIndexManager()


class Identificacion(BaseModel):
    id: int
    nombre: str
    identificacion: str


@router.get("/")
def read_identificaciones(search: str = None):
    if search:
        results, _ = manager.search(search)
        return results.to_dict(orient="records")
    else:
        data = manager.data
        return data.to_dict(orient="records")


@router.get("/{id}")
def read_identificacion(id: int):
    data = manager.data
    registro = data[data["id"] == id]
    return registro.to_dict(orient="records")


@router.post("/")
def create_identificacion(item: Identificacion):
    # Convertir el registro a DataFrame
    df_new = pd.DataFrame([item.model_dump()])
    # Insertar nuevos datos y sus embeddings
    manager.insert_data(df_new)
    return {"message": "Registro insertado", "data": item.model_dump()}


@router.put("/")
def update_identificacion(item: Identificacion):
    # Convertir el registro actualizado a DataFrame
    df_update = pd.DataFrame([item.model_dump()])
    # Actualizar el registro y el embedding correspondiente
    manager.update_data(df_update)
    return {"message": "Registro actualizado", "data": item.model_dump()}


@router.delete("/{id}")
def delete_identificacion(id: int):
    # Eliminar el registro y su embedding asociado
    manager.delete_data([id])
    return {"message": "Registro eliminado", "id": id}
