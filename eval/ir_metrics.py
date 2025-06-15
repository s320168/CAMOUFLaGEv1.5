from typing import Dict, List

from tqdm import tqdm


def _extract_id(query_id: str, ids) -> str:
    id = '.'.join(query_id.split('.')[:-1]) if '.' in query_id else query_id  # remove extension
    id = '-'.join(query_id.split('-12345')[:-1]) if '-12345' in id else id  # remove seed
    id = ':'.join(query_id.split('_12345')[:-1]) if '_12345' in id else id  # remove seed
    id = '_'.join(id.split('_0')[:-1]) if '_0' in id else id  # remove counter
    id = id.split('/')[-1]  # remove folder name
    if ids is not None:
        id = ids[id]
    return id


def precision_at_k(nn_map: Dict[str, List[str]], query_id: str, k: int, ids) -> float:
    id = _extract_id(query_id, ids)
    relevant_retireved_imgs = sum(1 for img in nn_map[query_id][:k] if id == _extract_id(img, ids))
    return float(relevant_retireved_imgs) / k


def recall_at_k(nn_map: Dict[str, List[str]], all_imgs: List[str], query_id: str, k: int, ids) -> float:
    id = _extract_id(query_id, ids)
    for img in nn_map[query_id][:k]:
        if id == _extract_id(img, ids):
            return 1
    return 0


def rel_at_k(nn_map: Dict[str, List[str]], query_id: str, k: int, ids) -> int:
    id = _extract_id(query_id, ids)
    return 1 if id == _extract_id(nn_map[query_id][k - 1], ids) else 0


def average_precision_at_k(nn_map: Dict[str, List[str]], all_imgs: List[str], query_id: str, k: int, ids) -> float:
    id = _extract_id(query_id, ids)
    gtp = sum(1 for img in all_imgs if id == _extract_id(img, ids))
    overall_ap = sum(
        precision_at_k(nn_map, query_id, n, ids) if rel_at_k(nn_map, query_id, n, ids) else 0 for n in range(1, k + 1))
    return overall_ap / gtp


def mean_average_precision_at_k(nn_map: Dict[str, List[str]], all_imgs: List[str], k: int, ids) -> float:
    overall_ap = sum(
        average_precision_at_k(nn_map, all_imgs, query_id, k, ids) for query_id in tqdm(nn_map.keys(), desc=f"mAP@{k}"))
    return overall_ap / len(nn_map.keys())


def mean_recall_at_k(nn_map: Dict[str, List[str]], all_imgs: List[str], k: int, ids) -> float:
    overall_recall_at_k = sum(
        recall_at_k(nn_map, all_imgs, query_id, k, ids) for query_id in tqdm(nn_map.keys(), desc=f"mR@{k}"))
    return overall_recall_at_k / len(nn_map.keys())
