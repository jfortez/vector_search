from enum import Enum


class SearchMode(str, Enum):
    FUZZY = "fuzzy"
    FAISS = "faiss"
