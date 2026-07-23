"""
data/image_dataset.py — ulcer crop 이미지 Dataset.

폴더 구조: image_root/{환자 id}/{visit}/{img2d 파일명}
    예) image_root/0019/1/photo_c.jpg   (환자 0019, 1차 재진, 사진 photo_c.jpg)

환자 한 명이 여러 방문(visit)을 가질 수 있고, 한 방문 안에서도 여러 장의 사진을 찍을 수
있다는 두 가지 전제를 폴더 구조에 반영했다. CSV의 id/visit/img2d 세 컬럼만 있으면 실제
파일 경로가 항상 유일하게 결정된다.

이 폴더 구조는 학습 시 환자 단위 GroupShuffleSplit(trainers/common.py, id 기준)과도
자연스럽게 맞아떨어진다 — 같은 환자의 모든 visit·이미지가 전부 같은 id 아래 있으므로,
split이 id 기준으로 이뤄지는 한 그 환자의 모든 방문 기록이 항상 같은 쪽(train 또는 val)에
남는다 (visit이 여러 개여도 split 단위는 여전히 id).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def letterbox_resize(image: Image.Image, target_size: int, fill_color=(0, 0, 0)) -> Image.Image:
    """정사각형으로 padding한 뒤 target_size x target_size로 리사이즈."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill_color)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((target_size, target_size), Image.BILINEAR)


def build_image_path(image_root: str | Path, patient_id: str, visit, img2d: str) -> Path:
    """image_root/{id}/{visit}/{파일명} 조합으로 실제 파일 경로를 만든다.

    `img2d`는 bare 파일명(`photo_c.jpg`)일 수도 있고, 이 프로젝트의 CSV처럼 예전 디렉토리
    구조 기준의 절대경로(`D:/dfu/toy_image_dataset/...`)가 그대로 들어있을 수도 있다 —
    두 경우 모두 `Path(img2d).name`으로 파일명만 뽑아 image_root 기준으로 다시 조립하면
    안전하다.
    """
    return Path(image_root) / str(patient_id) / str(int(visit)) / Path(str(img2d)).name


class WoundImageDataset(Dataset):
    """keys_df(반드시 'id', 'visit', 'img2d' 컬럼 포함)의 각 행에 대응하는 이미지를 로딩."""

    def __init__(
        self,
        keys_df: pd.DataFrame,
        image_root: str | Path,
        target_resolution: int,
        normalize_mean: tuple[float, float, float],
        normalize_std: tuple[float, float, float],
    ) -> None:
        for col in ("id", "visit", "img2d"):
            if col not in keys_df.columns:
                raise ValueError(f"keys_df에 '{col}' 컬럼이 없습니다.")
        self.keys_df = keys_df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.target_resolution = target_resolution
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])

    def __len__(self) -> int:
        return len(self.keys_df)

    def __getitem__(self, idx: int):
        row = self.keys_df.iloc[idx]
        patient_id, visit, img2d = str(row["id"]), row["visit"], str(row["img2d"])
        path = build_image_path(self.image_root, patient_id, visit, img2d)
        try:
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            warnings.warn(f"이미지를 열 수 없습니다: {path} ({exc}) — 이 샘플은 건너뜁니다.")
            return None

        image = letterbox_resize(image, self.target_resolution)
        tensor = self.transform(image)
        return tensor, patient_id, int(visit), img2d


def collate_skip_none(batch):
    """로딩 실패(None)한 샘플을 배치에서 제외."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), [], [], []
    tensors, ids, visits, img2ds = zip(*batch)
    return torch.stack(tensors), list(ids), list(visits), list(img2ds)
