from pydantic import BaseModel
from typing import Optional


class BaseList(BaseModel):
    CredencialId: int
    Identificacion: Optional[str]
    NombreCompleto: str
    FuenteId: int
    CargaId: Optional[int] = None
    FechaCarga: str
    Fuente: str
    Estado: str
