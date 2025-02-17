from fastapi import APIRouter
from embedding import searchFromIndex, data


router = APIRouter(prefix="/identificaciones", tags=["identificaciones"])


@router.get("/")
def read_identificaciones(search: str = None):
    if search:
        result = searchFromIndex(search)
        return result[0].to_dict(orient="records")
    else:
        return data.to_dict(orient="records")


@router.get("/{id}")
def read_identificacion(id: int):
    data = data[data['id'] == id]
    return data.to_dict(orient="records")
