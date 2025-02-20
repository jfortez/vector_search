import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from io import StringIO
import os
import pandas as pd
from time import time


class FaissManager:
    def __init__(self, embedding_dim, index_type="L2", nlist=50):
        """
        Initialize FaissManager with the specified index type.

        Args:
            embedding_dim (int): Dimension of the embeddings
            index_type (str): Type of FAISS index ('L2', 'IVFFlat', 'IVFPQ')
            nlist (int): Number of clusters for IVF indices (default=50)
        """
        self.d = embedding_dim  # Dimension of embeddings
        self.index_type = index_type.lower()
        self.nlist = nlist

        # Initialize the index based on the type
        self.index = self._create_index()

    def _create_index(self):
        """Create the appropriate FAISS index based on index_type."""
        if self.index_type == "l2":
            # Flat L2 index (exact search)
            index = faiss.IndexFlatL2(self.d)
        elif self.index_type == "ivfflat":
            # Partitioning with IVF (Inverted File) and Flat L2 quantizer
            quantizer = faiss.IndexFlatL2(self.d)
            index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
        elif self.index_type == "ivfpq":
            # Product Quantizer with IVF
            quantizer = faiss.IndexFlatL2(self.d)
            index = faiss.IndexIVFPQ(
                quantizer, self.d, self.nlist, 8, 8
            )  # 8 bits per subquantizer
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        return index

    def load_embeddings(self, npy_files_dir):
        """
        Load embeddings from .npy files in the specified directory.

        Args:
            npy_files_dir (str): Directory containing .npy embedding files
        """

        # Get all .npy files
        emb_files = [f for f in os.listdir(npy_files_dir) if f.endswith(".npy")]

        # Sort files numerically based on the number in the filename
        emb_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        # Initialize an empty list to store the embeddings
        embeddings = []
        for file in emb_files:
            emb = np.load(os.path.join(npy_files_dir, file))
            embeddings.append(emb)  # Append the entire NumPy array

        self.embeddings = np.vstack(embeddings)
        # self.embeddings = np.concatenate(embeddings, axis=0)

        print(
            f"Loaded {self.embeddings.shape[0]} embeddings with dim {self.embeddings.shape[1]}"
        )

        # Add embeddings to the index

        if self.index_type in ["ivfflat", "ivfpq"]:
            print("[INIT] Training index...")
            # Train the index if using IVF-based indices
            self.index.train(self.embeddings)

        self.index.add(self.embeddings)
        print("[INIT] Index added to the index.")

    def search(self, query_embedding, k=5):
        """
        Search for the k nearest neighbors of a query embedding.

        Args:
            query_embedding (np.array): Query embedding
            k (int): Number of nearest neighbors to return

        Returns:
            tuple: (distances, indices) of the k nearest neighbors
        """
        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]

    def save_index(self, filepath):
        """Save the FAISS index to a file."""
        faiss.write_index(self.index, filepath)

    def load_index(self, filepath):
        """Load a FAISS index from a file."""
        self.index = faiss.read_index(filepath)


class DataManager:
    def __init__(self, path: str):
        self.path = path if path else "sentences.txt"
        self.dataset = []
        # validate if path exist and is a valid .txt file
        if os.path.exists(path) and os.path.isfile(path):
            self.load_dataset(path)
        else:
            self.download_dataset()

    def download_dataset(self):
        url = [
            "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.train.tsv",
            "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/MSRpar.test.tsv",
            "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2012/OnWN.test.tsv",
            "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2013/OnWN.test.tsv",
            "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/OnWN.test.tsv",
            "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2014/images.test.tsv",
            "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/semeval-sts/2015/images.test.tsv",
        ]
        res = requests.get(
            "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt"
        )
        text = res.text
        dataset = pd.read_csv(StringIO(text), sep="\t", on_bad_lines="skip")

        sentences = dataset["sentence_A"].tolist()
        sentences_b = dataset["sentence_B"].tolist()
        sentences.extend(sentences_b)

        for url in url:
            res = requests.get(url)
            text = res.text
            dataset = pd.read_csv(
                StringIO(text), sep="\t", on_bad_lines="skip", header=None
            )
            sentences.extend(dataset[1].tolist())
            sentences.extend(dataset[2].tolist())

        # remove duplicates and NaN
        sentences = [
            sentence.replace("\n", "")
            for sentence in list(set(sentences))
            if type(sentence) is str
        ]
        self.dataset = sentences
        self.save_dataset()

    def get_dataset(self):
        return self.dataset

    def save_dataset(self):
        # save the dataset to a .txt file
        file = open(self.path, "w")
        for sentence in self.dataset:
            file.write(sentence + "\n")

    def load_dataset(self, path: str):
        file = open(path, "r")
        self.dataset = pd.read_csv(file, sep="\t", on_bad_lines="skip", header=None)[
            0
        ].tolist()


if __name__ == "__main__":
    # Example usage
    EMBEDDING_DIR = "notebooks/sim_sentences"
    EMBEDDING_DIM = 768
    data_manager = DataManager("notebooks/sentences.txt")
    data = data_manager.get_dataset()

    # Initialize with different index types
    index_types = ["l2", "ivfflat", "ivfpq"]

    # Example: Generate a query embedding using SentenceTransformer
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    query_text = "Want to play football"
    query_embedding = model.encode(query_text)

    print(f"[QUERY]:  {query_text}")
    print(f"DATASET LENGTH: {len(data)}")
    print("-" * 28)
    for idx_type in index_types:
        t = time.time()
        print(f"\nTesting with {idx_type.upper()} index...")
        print("-" * 28)

        # Create FaissManager instance
        faiss_mgr = FaissManager(embedding_dim=EMBEDDING_DIM, index_type=idx_type)

        print(f"Is Trained? {faiss_mgr.index.is_trained}")

        # Load embeddings from .npy files
        faiss_mgr.load_embeddings(EMBEDDING_DIR)

        # Search for top 5 similar embeddings
        distances, indices = faiss_mgr.search(query_embedding, k=5)
        max_distance = distances.max() if distances.max() != 0 else 1

        # # Print results in a structured format
        results_df = pd.DataFrame(
            [
                {
                    # "Neighbor": i + 1,
                    "Idx": indices[i],
                    "Sentence": data[indices[i]],
                    "Similarity": f"{(1 - distances[i] / max_distance):.2%}",
                }
                for i in range(len(distances))
            ]
        )
        print(results_df)
        t = time.time() - t
        print(f"Time: {t:.4f} seconds")

    # Save the index for later use
    # faiss_mgr.save_index(f"index_{idx_type}.faiss")
