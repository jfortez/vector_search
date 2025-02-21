from fastapi import APIRouter, Query
from models.identificacion import Identificacion, IdentificacionCreate
from models.search import SearchMode
from typing import Optional, List

from api.exceptions import NotFoundException, BadRequestException
from search.main import SearchManager


router = APIRouter(prefix="/identificaciones", tags=["identificaciones"])

# Modelo Pydantic para los registros

manager = SearchManager()


@router.get(
    "/",
    response_model=List[Identificacion],
    summary="Consulta de Identificaciones",
    status_code=200,
    description="Consulta o Busqueda de Identificaciones",
)
def get_identificaciones(
    search: Optional[str] = Query(None, description="Término de búsqueda"),
    mode: SearchMode = Query(SearchMode.FAISS, description="Modo de búsqueda"),
    threshold: float = Query(0.1, description="Umbral de distancia"),
):
    try:
        if search:
            results = manager.search(search, mode, threshold)
        else:
            results = manager.dao.get_all()
        print(results)
        return results.to_dict(orient="records")
    except Exception as e:
        raise BadRequestException(f"Error en la búsqueda: {str(e)}")


@router.get(
    "/{id}",
    response_model=Identificacion,
    status_code=200,
    summary="Obtener identificación",
    description="Obtiene una identificación por su ID",
)
async def get_identificacion(id: int):
    data = manager.dao.find_by_id(id)

    if data.id is None:
        raise NotFoundException(f"Identificación con ID {id} no encontrada")

    return data.model_dump()


@router.post(
    "/",
    status_code=200,
    summary="Crear identificación",
    description="Crea una nueva identificación",
)
def create_identificacion(new_row: IdentificacionCreate):
    # Insertar nuevos datos y sus embeddings
    manager.insert_data(new_row)
    return {"message": "Registro insertado", "data": new_row.model_dump()}


@router.put(
    "/{id}",
    status_code=200,
    summary="Actualizar identificación",
    description="Actualiza una identificación existente",
)
def update_identificacion(item: Identificacion):
    # Actualizar el registro y el embedding correspondiente
    manager.update_data(item)
    return {"message": "Registro actualizado", "data": item.model_dump()}


@router.delete(
    "/{id}",
    status_code=200,
    summary="Eliminar identificación",
    description="Elimina una identificación existente",
)
def delete_identificacion(id: int):
    # Eliminar el registro y su embedding asociado
    manager.delete_data(id)
    return {"message": "Registro eliminado", "id": id}
