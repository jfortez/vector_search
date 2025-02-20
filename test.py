from search.fuzzy_search import FuzzyManager


if __name__ == "__main__":
    manager = FuzzyManager()
    print(manager.search("Juan"))
