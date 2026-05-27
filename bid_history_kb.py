"""
历史招投标文件知识库（基于 TF-IDF 向量检索）
完全本地化，无需网络，不依赖 sentence-transformers / faiss
"""

import os
import pickle
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HistoricalBidKB:
    def __init__(self, storage_dir="E:/hello-agents/code/chapter1/knowledge/bid_history"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.index_path = os.path.join(storage_dir, "tfidf_index.pkl")
        self.metadata_path = os.path.join(storage_dir, "metadata.pkl")
        
        self.vectorizer = None
        self.tfidf_matrix = None
        self.metadata = []
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            with open(self.index_path, 'rb') as f:
                self.vectorizer, self.tfidf_matrix = pickle.load(f)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.vectorizer = TfidfVectorizer(stop_words=None, max_features=5000)
            self.tfidf_matrix = None
            self.metadata = []
    
    def add_document(self, filename: str, snippets: List[str], opinions: List[str]):
        if len(snippets) != len(opinions):
            raise ValueError("snippets 和 opinions 长度必须一致")
        
        valid = [(s, o) for s, o in zip(snippets, opinions) if s.strip()]
        if not valid:
            return
        
        new_snippets, new_opinions = zip(*valid)
        new_metadata = [
            {"filename": filename, "snippet": s, "opinion": o}
            for s, o in zip(new_snippets, new_opinions)
        ]
        self.metadata.extend(new_metadata)
        
        corpus = [item["snippet"] for item in self.metadata]
        if len(corpus) > 0:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        else:
            self.tfidf_matrix = None
        
        self._save()
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.metadata or self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    "filename": self.metadata[idx]["filename"],
                    "snippet": self.metadata[idx]["snippet"],
                    "opinion": self.metadata[idx]["opinion"],
                    "score": float(similarities[idx])
                })
        return results
    
    def _save(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump((self.vectorizer, self.tfidf_matrix), f)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def clear_all(self):
        self.metadata = []
        self.vectorizer = TfidfVectorizer(stop_words=None, max_features=5000)
        self.tfidf_matrix = None
        self._save()