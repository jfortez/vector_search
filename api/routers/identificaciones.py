from fastapi import APIRouter, Query, Depends
from search.faiss_search import FaissIndexManager
from search.fuzzy_search import FuzzyManager
from models.identificacion import Identificacion, IdentificacionCreate
from typing import Optional, List
from enum import Enum
from api.exceptions import NotFoundException, BadRequestException


router = APIRouter(prefix="/identificaciones", tags=["identificaciones"])

# Modelo Pydantic para los registros

faiss_manager = FaissIndexManager()
fuzz_manager = FuzzyManager()


class SearchMode(str, Enum):
    FUZZY = "fuzzy"
    FAISS = "faiss"


@router.get(
    "/",
    response_model=List[Identificacion],
    summary="Consulta de Identificaciones",
    description="Consulta o Busqueda de Identificaciones",
)
def get_identificaciones(
    search: Optional[str] = Query(None, description="Término de búsqueda"),
    mode: SearchMode = Query(SearchMode.FAISS, description="Modo de búsqueda"),
):
    try:
        if search:
            if mode == SearchMode.FAISS:
                r, _ = faiss_manager.search(search)
                results = r
            else:
                results = fuzz_manager.search(search, threshold=70)

        else:
            results = faiss_manager.db.get_all()
        return results.to_dict(orient="records")
    except Exception as e:
        raise BadRequestException(f"Error en la búsqueda: {str(e)}")


@router.get(
    "/{id}",
    response_model=Identificacion,
    summary="Obtener identificación",
    description="Obtiene una identificación por su ID",
)
async def get_identificacion(id: int):
    data = faiss_manager.db.find_by_id(id)

    if data.id is None:
        raise NotFoundException(f"Identificación con ID {id} no encontrada")

    return data.model_dump()


@router.post(
    "/",
    status_code=201,
    summary="Crear identificación",
    description="Crea una nueva identificación",
)
def create_identificacion(item: IdentificacionCreate):
    # Insertar nuevos datos y sus embeddings
    faiss_manager.insert_data(item.identificacion, item.nombre)
    return {"message": "Registro insertado", "data": item.model_dump()}


@router.put(
    "/{id}",
    summary="Actualizar identificación",
    description="Actualiza una identificación existente",
)
def update_identificacion(item: Identificacion):
    # Actualizar el registro y el embedding correspondiente
    faiss_manager.update_data(item)
    return {"message": "Registro actualizado", "data": item.model_dump()}


@router.delete(
    "/{id}",
    status_code=204,
    summary="Eliminar identificación",
    description="Elimina una identificación existente",
)
def delete_identificacion(id: int):
    # Eliminar el registro y su embedding asociado
    faiss_manager.delete_data(id)
    return {"message": "Registro eliminado", "id": id}
