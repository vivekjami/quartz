import math
import random
import struct
from pathlib import Path
import numpy as np

class HNSWGraph:
    """
    Simplified HNSW graph (Malkov & Yashunin, 2018).
    2 layers: base layer (all nodes) + top layer (random ~5%).

    Key invariants:
    - Each node at layer l has at most M bidirectional neighbors
    - Entry point is the node with the highest layer assignment
    - Search starts at entry_point and greedily descends layers
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200):
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2  # base layer allows more neighbors
        self.ef_construction = ef_construction
        self.ml = 1.0 / math.log(M)  # level normalization factor

        self.vectors: dict[int, np.ndarray] = {}   # node_id → f16 vector
        self.layers: list[dict[int, list[int]]] = []  # layer → {node: [neighbors]}
        self.entry_point: int | None = None
        self.max_layer: int = 0

    def _random_level(self) -> int:
        """Assign a random level to a new node. Level 0 is most common."""
        return int(-math.log(random.random()) * self.ml)

    def _cosine_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance (1 - similarity). Operates on f32 promoted from f16."""
        a32 = a.astype(np.float32)
        b32 = b.astype(np.float32)
        dot = np.dot(a32, b32)
        norm = np.linalg.norm(a32) * np.linalg.norm(b32)
        if norm == 0:
            return 1.0
        return float(1.0 - dot / norm)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_ids: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """
        Greedy beam search within a single layer.
        Returns ef nearest neighbors as (distance, node_id) sorted ascending.
        """
        import heapq
        visited = set(entry_ids)
        candidates = []   # min-heap: (dist, node_id)
        results = []      # max-heap: (-dist, node_id) — we want to eject the worst

        for eid in entry_ids:
            d = self._cosine_dist(query, self.vectors[eid].astype(np.float32))
            heapq.heappush(candidates, (d, eid))
            heapq.heappush(results, (-d, eid))

        while candidates:
            dist, node = heapq.heappop(candidates)
            # Termination: if the closest candidate is worse than our ef-th result, stop
            if results and dist > -results[0][0]:
                break

            layer_graph = self.layers[layer] if layer < len(self.layers) else {}
            for neighbor in layer_graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    nd = self._cosine_dist(query, self.vectors[neighbor].astype(np.float32))
                    heapq.heappush(candidates, (nd, neighbor))
                    heapq.heappush(results, (-nd, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)  # eject the farthest

        return sorted((-d, n) for d, n in results)

    def add(self, node_id: int, vector: np.ndarray):
        """Add a node to the graph. vector must be shape (dim,)."""
        # Store as f16 to halve memory
        self.vectors[node_id] = vector.astype(np.float16)
        level = self._random_level()

        # Extend layers list if needed
        while len(self.layers) <= level:
            self.layers.append({})

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = level
            for l in range(level + 1):
                self.layers[l][node_id] = []
            return

        # Phase 1: greedy descent from max_layer to level+1 (find entry for insertion)
        ep = [self.entry_point]
        for l in range(self.max_layer, level, -1):
            ep = [n for _, n in self._search_layer(vector, ep, ef=1, layer=l)]

        # Phase 2: insert at each layer from min(level, max_layer) to 0
        for l in range(min(level, self.max_layer), -1, -1):
            neighbors_found = self._search_layer(vector, ep, ef=self.ef_construction, layer=l)
            M_at_layer = self.M_max0 if l == 0 else self.M
            # Select M closest neighbors (simple heuristic — production HNSW uses HNSW-selection)
            selected = [n for _, n in neighbors_found[:M_at_layer]]

            self.layers[l].setdefault(node_id, []).extend(selected)
            # Add back edges
            for neighbor in selected:
                self.layers[l].setdefault(neighbor, []).append(node_id)
                # Prune if neighbor has too many connections
                if len(self.layers[l][neighbor]) > M_at_layer:
                    # Keep M_at_layer closest neighbors
                    nv = self.vectors[neighbor].astype(np.float32)
                    scored = [
                        (self._cosine_dist(nv, self.vectors[n].astype(np.float32)), n)
                        for n in self.layers[l][neighbor]
                    ]
                    scored.sort()
                    self.layers[l][neighbor] = [n for _, n in scored[:M_at_layer]]

            ep = [n for _, n in neighbors_found[:1]]

        # Update entry point if this node has higher level
        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = node_id

    def search(self, query: np.ndarray, k: int, ef: int = 50) -> list[tuple[float, int]]:
        """
        Search for k nearest neighbors.
        ef controls beam width: higher ef = better recall, slower query.
        The ef vs recall tradeoff is the key HNSW tuning parameter.
        """
        if self.entry_point is None:
            return []

        ep = [self.entry_point]
        for l in range(self.max_layer, 0, -1):
            ep = [n for _, n in self._search_layer(query, ep, ef=1, layer=l)]

        results = self._search_layer(query, ep, ef=max(ef, k), layer=0)
        return results[:k]

    def save(self, path: Path):
        """Serialize graph to disk."""
        import pickle
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "graph.pkl", "wb") as f:
            pickle.dump({
                "layers": self.layers,
                "entry_point": self.entry_point,
                "max_layer": self.max_layer,
                "M": self.M,
                "dim": self.dim,
            }, f, protocol=5)
        # Vectors saved separately as a numpy array for efficient mmap
        ids = list(self.vectors.keys())
        vecs = np.stack([self.vectors[i] for i in ids])  # (N, dim), f16
        np.save(str(path / "vectors.npy"), vecs)
        np.save(str(path / "vector_ids.npy"), np.array(ids, dtype=np.int64))

    @classmethod
    def load(cls, path: Path) -> "HNSWGraph":
        import pickle
        with open(path / "graph.pkl", "rb") as f:
            state = pickle.load(f)
        g = cls(dim=state["dim"], M=state["M"])
        g.layers = state["layers"]
        g.entry_point = state["entry_point"]
        g.max_layer = state["max_layer"]
        ids = np.load(str(path / "vector_ids.npy"))
        vecs = np.load(str(path / "vectors.npy"))
        for i, node_id in enumerate(ids):
            g.vectors[int(node_id)] = vecs[i]
        return g
