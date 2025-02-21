from fastapi import APIRouter
from typing import List
import requests
from fastapi.responses import JSONResponse
from models.list import BaseList
from ..exceptions import NotFoundException


router = APIRouter(prefix="/list", tags=["list"])


@router.get(
    "/",
    response_model=List[BaseList],
    status_code=200,
    summary="Consulta de Listas",
    description="Consulta o Busqueda de Listas",
)
def get_list():

    url = "http://172.16.11.132:4000/api/getProcessedData"

    try:
        response = requests.get(url)

        response.raise_for_status()

        content = response.json()

        return JSONResponse(content)

    except requests.exceptions.RequestException as e:
        raise NotFoundException(f"Error: {e}")
