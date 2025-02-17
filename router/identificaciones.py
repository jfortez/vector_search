from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

from embedding import (
    searchFromIndex,
    get_data,
    insert_data,
    update_data,
    delete_data
)

router = APIRouter(prefix="/identificaciones", tags=["identificaciones"])

# Modelo Pydantic para los registros


class Identificacion(BaseModel):
    id: int
    nombre: str
    identificacion: str


@router.get("/")
def read_identificaciones(search: str = None):
    if search:
        results, _ = searchFromIndex(search)
        return results.to_dict(orient="records")
    else:
        data = get_data()
        return data.to_dict(orient="records")


@router.get("/{id}")
def read_identificacion(id: int):
    data = get_data()
    registro = data[data['id'] == id]
    return registro.to_dict(orient="records")


@router.post("/")
def create_identificacion(item: Identificacion):
    # Convertir el registro a DataFrame
    df_new = pd.DataFrame([item.model_dump()])
    # Insertar nuevos datos y sus embeddings
    insert_data(df_new)
    return {"message": "Registro insertado", "data": item.dict()}


@router.put("/")
def update_identificacion(item: Identificacion):
    # Convertir el registro actualizado a DataFrame
    df_update = pd.DataFrame([item.model_dump()])
    # Actualizar el registro y el embedding correspondiente
    update_data(df_update)
    return {"message": "Registro actualizado", "data": item.dict()}


@router.delete("/{id}")
def delete_identificacion(id: int):
    # Eliminar el registro y su embedding asociado
    delete_data([id])
    return {"message": "Registro eliminado", "id": id}
