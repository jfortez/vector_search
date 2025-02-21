from database.connection import Database
from database.dao.identificacion import IdentificacionDAO
from .faiss_search import FaissIndexManager
from .fuzzy_search import FuzzyManager
from models.search import SearchMode
import pandas as pd


class SearchManager:
    def __init__(self):
        db = Database()
        dao = IdentificacionDAO(db)
        self.dao = dao

        self.data = dao.get_all().copy()
        self.faiss_manager = FaissIndexManager(dao, self.data)
        self.fuzzy_manager = FuzzyManager(dao, self.data)

    def search(
        self, query: str, mode: SearchMode = SearchMode.FAISS, threshold: float = 0.1
    ) -> pd.DataFrame:
        if mode == SearchMode.FAISS:
            r, _ = self.faiss_manager.search(query, threshold)
            return r
        elif mode == SearchMode.FUZZY:
            return self.fuzzy_manager.search(query, threshold)

    def insert_data(self, new_row: dict):
        self.faiss_manager.insert_data(new_row)
        self.fuzzy_manager.data = self.faiss_manager.data

    def update_data(self, row: dict):
        self.faiss_manager.update_data(row)
        self.fuzzy_manager.data = self.faiss_manager.data

    def delete_data(self, id: int):
        self.faiss_manager.delete_data(id)
        self.fuzzy_manager.data = self.faiss_manager.data
