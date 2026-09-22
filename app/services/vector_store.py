import faiss
import numpy as np
import os
import json
from typing import Tuple, List, Dict, Any

class VectorStore:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.flat_index = None
        self.ivf_index = None
        self.metadata = {}  # Map faiss ID to metadata
        self.current_id = 0

    def create_index(self, index_type: str = "flat", nlist: int = 50):
        if index_type == "flat":
            self.flat_index = faiss.IndexFlatL2(self.dimension)
            # Wrap in IndexIDMap to support add_with_ids
            self.flat_index = faiss.IndexIDMap(self.flat_index)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.ivf_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

    def add_vectors(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]], index_type: str = "flat"):
        if vectors.shape[0] == 0:
            return
            
        ids = np.arange(self.current_id, self.current_id + vectors.shape[0])
        
        for i, meta in enumerate(metadatas):
            self.metadata[int(ids[i])] = meta

        if index_type == "flat":
            if self.flat_index is None:
                self.create_index("flat")
            self.flat_index.add_with_ids(vectors, ids)
            
        elif index_type == "ivf":
            if self.ivf_index is None:
                self.create_index("ivf")
                
            if not self.ivf_index.is_trained:
                train_vectors = vectors
                if vectors.shape[0] < self.ivf_index.nlist:
                    repeats = (self.ivf_index.nlist // vectors.shape[0]) + 1
                    train_vectors = np.tile(vectors, (repeats, 1))
                self.ivf_index.train(train_vectors)
                
            self.ivf_index.add_with_ids(vectors, ids)
            
        self.current_id += vectors.shape[0]

    def search(self, query_vector: np.ndarray, k: int = 10, index_type: str = "flat") -> Tuple[List[Dict[str, Any]], List[float]]:
        if index_type == "flat" and self.flat_index:
            distances, indices = self.flat_index.search(query_vector, k)
        elif index_type == "ivf" and self.ivf_index:
            self.ivf_index.nprobe = min(10, self.ivf_index.nlist)
            distances, indices = self.ivf_index.search(query_vector, k)
        else:
            return [], []

        results = []
        result_distances = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and int(idx) in self.metadata:
                results.append(self.metadata[int(idx)])
                result_distances.append(float(distances[0][i]))
                
        return results, result_distances

    def save_index(self, path: str, index_type: str = "flat"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if index_type == "flat" and self.flat_index:
            # IndexIDMap needs to be cast back or written directly
            faiss.write_index(self.flat_index, f"{path}_flat.index")
        elif index_type == "ivf" and self.ivf_index:
            faiss.write_index(self.ivf_index, f"{path}_ivf.index")
            
        with open(f"{path}_metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def load_index(self, path: str, index_type: str = "flat"):
        if index_type == "flat" and os.path.exists(f"{path}_flat.index"):
            self.flat_index = faiss.read_index(f"{path}_flat.index")
        elif index_type == "ivf" and os.path.exists(f"{path}_ivf.index"):
            self.ivf_index = faiss.read_index(f"{path}_ivf.index")
            
        metadata_path = f"{path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                loaded_meta = json.load(f)
                self.metadata = {int(k): v for k, v in loaded_meta.items()}
                if self.metadata:
                    self.current_id = max(self.metadata.keys()) + 1
