# index_manager.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from database.connection import Database
from database.dao.identificacion import IdentificacionDAO
from util import normalize
from models.identificacion import IdentificacionCreate, Identificacion


class BaseIndexManager:
    """
    Clase base para administrar índices de búsqueda.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el modelo y carga datos iniciales desde la base de datos.
        """
        self.model = SentenceTransformer(model_name)
        self.db = IdentificacionDAO(Database())
        self.data = self.db.get_all().copy()
        self.embeddings = self._create_embeddings(self.data)

    def _create_embeddings(self, df: pd.DataFrame) -> dict:
        """
        Crea un diccionario de embeddings para búsqueda rápida.
        """
        texts = df.apply(
            lambda row: normalize(f"{row['nombre']} {row['identificacion']}"), axis=1
        ).tolist()
        return {idx: text for idx, text in zip(df["id"], texts)}

    def search(self, query: str, threshold: float, k: int):
        """
        Método abstracto para búsqueda. Debe ser implementado en subclases.
        """
        raise NotImplementedError("Este método debe ser implementado por subclases.")

    def _revalidate_data(self):
        """
        Actualiza los datos en memoria desde la base de datos.
        """
        self.data = self.db.get_all().copy()

    def insert_data(self, new_row: IdentificacionCreate):
        """
        Inserta nuevos registros en el índice.
        """
        self.db.insert(new_row)
        self._revalidate_data()

    def delete_data(self, id: int):
        """
        Elimina registros del índice.
        """
        self.db.delete(id)
        self._revalidate_data()

    def update_data(self, row: Identificacion):
        """
        Actualiza registros en la base de datos.
        """
        self.db.update(row)
        self._revalidate_data()
