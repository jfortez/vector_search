from pydantic import BaseModel, Field


class IdentificacionBase(BaseModel):
    nombre: str = Field(..., min_length=2, max_length=100)
    identificacion: str = Field(..., min_length=5, max_length=50)


class IdentificacionCreate(IdentificacionBase):
    pass


class Identificacion(IdentificacionBase):
    id: int

    class Config:
        from_attributes = True
