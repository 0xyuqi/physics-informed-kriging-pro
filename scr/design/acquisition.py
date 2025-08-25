from __future__ import annotations
from typing import Iterable, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist


def _enforce_min_dist(chosen_coords: List[np.ndarray], cand_coords: np.ndarray, min_dist: float) -> np.ndarray:
    
    if not chosen_coords:
        return np.ones(len(cand_coords), dtype=bool)
    D = cdist(np.vstack(chosen_coords), cand_coords)
    mask = (D >= float(min_dist)).all(axis=0)
    return mask


def select_by_variance(model, X_candidates: np.ndarray, k: int, min_dist: float = 0.0) -> List[int]:
    
    X_candidates = np.asarray(X_candidates)
    chosen_idx: List[int] = []
    chosen_coords: List[np.ndarray] = []

    remaining = np.arange(len(X_candidates))
    for _ in range(k):
       
        mask = _enforce_min_dist([X_candidates[i] for i in chosen_idx], X_candidates[remaining], min_dist)
        valid = remaining[mask]
        if len(valid) == 0:
            break
    
        _, std = model.predict(X_candidates[valid], return_std=True)
        pick_local = int(valid[np.argmax(std)])
        chosen_idx.append(pick_local)
        
        remaining = np.array([i for i in remaining if i != pick_local], dtype=int)
    return chosen_idx


def select_by_a_optimal(model, X_candidates: np.ndarray, grid_eval: np.ndarray, k: int, min_dist: float = 0.0) -> List[int]:
   
    X_candidates = np.asarray(X_candidates)
    grid_eval = np.asarray(grid_eval)

    chosen_idx: List[int] = []
    remaining = np.arange(len(X_candidates))

    _, std0 = model.predict(grid_eval, return_std=True)
    base_sum = float((std0 ** 2).sum())

    for _ in range(k):
        best_i, best_drop = None, -np.inf
       
        mask = _enforce_min_dist([X_candidates[i] for i in chosen_idx], X_candidates[remaining], min_dist)
        valid = remaining[mask]
        if len(valid) == 0:
            break
        for i in valid:
           
            x_i = X_candidates[i].reshape(1, -1)
            
            d2 = ((grid_eval - x_i) ** 2).sum(axis=1)
            w = np.exp(-d2 / (2.0 * 3.0 ** 2))  
            drop = float((w * std0 ** 2).sum())
            if drop > best_drop:
                best_drop, best_i = drop, int(i)
        if best_i is None:
            break
        chosen_idx.append(best_i)
        
        remaining = np.array([i for i in remaining if i != best_i], dtype=int)
    return chosen_idx