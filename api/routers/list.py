from fastapi import APIRouter
from typing import List
from models.list import BaseList
from search.lists import EmbeddingManager
from fastapi.responses import JSONResponse

manager = EmbeddingManager()


router = APIRouter(prefix="/list", tags=["list"])


@router.get(
    "/",
    response_model=List[BaseList],
    status_code=200,
    summary="Consulta de Listas",
    description="Consulta o Busqueda de Listas",
)
def get_list(search: str = None, threshold: float = 0.1, k: int = 10):
    data = manager.data[:1000]
    if search:
        data = manager.search(search, threshold, k)
        return data.to_dict(orient="records")
    return JSONResponse(content=data, status_code=200)
