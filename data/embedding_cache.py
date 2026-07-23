"""
data/embedding_cache.py — DINOv3 embedding을 디스크에 캐싱. 캐시 키는 (id, visit, img2d)
3개다 — 다회차 방문 구조에서는 같은 img2d 파일명이 (이론상) 다른 visit 폴더에 존재할 수
있어, visit까지 포함해야 진짜 유일한 키가 된다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import ImageEncoderConfig
from data.image_dataset import WoundImageDataset, collate_skip_none
from models.dinov3_backbone import DINOv3ImageEncoder

_KEY_COLS = ["id", "visit", "img2d"]


def _load_cache(cache_path: str | Path | None) -> pd.DataFrame:
    if cache_path is None:
        return pd.DataFrame(columns=_KEY_COLS)
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return pd.DataFrame(columns=_KEY_COLS)
    return pd.read_parquet(cache_path)


def _extract_for_keys(
    keys_df: pd.DataFrame,
    image_root: str | Path,
    config: ImageEncoderConfig,
    batch_size: int,
    num_workers: int,
    encoder: DINOv3ImageEncoder | None = None,
) -> pd.DataFrame:
    dataset = WoundImageDataset(
        keys_df, image_root, config.input_resolution, config.normalize_mean, config.normalize_std,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_skip_none)

    if encoder is None:
        encoder = DINOv3ImageEncoder(config)
    encoder.to(config.device)
    encoder.eval()

    all_emb, all_id, all_visit, all_img2d = [], [], [], []
    for tensors, ids, visits, img2ds in loader:
        if not ids:
            continue
        tensors = tensors.to(config.device)
        emb = encoder(tensors)
        all_emb.append(emb.cpu())
        all_id.extend(ids)
        all_visit.extend(visits)
        all_img2d.extend(img2ds)

    if not all_emb:
        raise RuntimeError("인코딩된 이미지가 하나도 없습니다 — 전부 로딩 실패했을 수 있습니다.")

    emb_tensor = torch.cat(all_emb, dim=0)
    result = pd.DataFrame(emb_tensor.numpy(), columns=[f"emb_{i}" for i in range(emb_tensor.shape[1])])
    result.insert(0, "img2d", all_img2d)
    result.insert(0, "visit", all_visit)
    result.insert(0, "id", all_id)
    return result


def get_or_extract_embeddings(
    df: pd.DataFrame,
    image_root: str | Path,
    config: ImageEncoderConfig,
    cache_path: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    encoder: DINOv3ImageEncoder | None = None,
) -> pd.DataFrame:
    """df(반드시 id/visit/img2d 포함)가 필요로 하는 모든 이미지의 embedding을 캐시 우선으로 확보."""
    required_keys = df[_KEY_COLS].dropna().drop_duplicates().reset_index(drop=True)
    required_keys["visit"] = required_keys["visit"].astype(int)

    cache = _load_cache(cache_path)
    if len(cache) > 0:
        cache["visit"] = cache["visit"].astype(int)

    merged_check = required_keys.merge(cache[_KEY_COLS] if len(cache) else cache, on=_KEY_COLS, how="left", indicator=True)
    missing_keys = merged_check[merged_check["_merge"] == "left_only"][_KEY_COLS].reset_index(drop=True)

    if len(missing_keys) > 0:
        print(f"[embedding_cache] 캐시에 없는 이미지 {len(missing_keys)}개 새로 인코딩")
        new_embeddings = _extract_for_keys(missing_keys, image_root, config, batch_size, num_workers, encoder)
        cache = pd.concat([cache, new_embeddings], ignore_index=True)
    else:
        print(f"[embedding_cache] 필요한 이미지 {len(required_keys)}개 전부 캐시에서 로딩 (재인코딩 없음)")

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path, index=False)
        print(f"[embedding_cache] 캐시 저장: {cache_path} (총 {len(cache)}개 이미지)")

    result = required_keys.merge(cache, on=_KEY_COLS, how="left")
    emb_cols = [c for c in cache.columns if c.startswith("emb_")]
    still_missing = result[result[emb_cols].isna().any(axis=1)]
    if len(still_missing) > 0:
        raise RuntimeError(
            f"{len(still_missing)}개 이미지의 embedding을 확보하지 못했습니다 (로딩 실패 등): "
            f"{still_missing[_KEY_COLS].values.tolist()[:5]}..."
        )

    return result
