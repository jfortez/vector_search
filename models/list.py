from pydantic import BaseModel


class BaseList(BaseModel):
    CredencialId: int
    Identificacion: str | None
    NombreCompleto: str
    FuenteId: int
    CargaId: int
    FechaCarga: str
    Fuente: str
    Estado: str
