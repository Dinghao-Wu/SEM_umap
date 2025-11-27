#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sem_fft_pipeline.py

機能概要：
  - SEM 画像に対して以下の処理を行うパイプライン：
      1) グレースケール化
      2) リサイズ（size x size）
      3) オプション：3x3 メディアンフィルタによるノイズ除去
      4) オプション：CLAHE による局所コントラスト強調
      5) オプション：画素強度の正規化（(x - mean) / std）
      6) オプション：2D Hann 窓の適用
      7) 2D FFT → 振幅二乗 |F|^2
      8) オプション：log10(|F|^2 + eps)
      9) オプション：パワースペクトルの標準化（(x - mean) / std）
  - 計算した 2D パワースペクトルを .npy と .png として保存
  - モジュールとして import してもよし、CLI から単体で実行してもよし
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from skimage.filters import median
from skimage.morphology import square


# 対応する画像拡張子
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ===================== 設定用 dataclass =====================

@dataclass
class SemFftConfig:
    """
    SEM → FFT パイプラインの各種パラメータ設定
    """
    size: int = 256              # 画像をリサイズするサイズ（size x size）
    use_median: bool = True      # 3x3 メディアンフィルタでのノイズ除去を行うか
    use_clahe: bool = False      # CLAHE による局所コントラスト強調を行うか
    use_intensity_norm: bool = True   # 入力画像の強度を (x - mean) / std で正規化するか
    use_window: bool = True      # 2D Hann 窓を適用するか
    use_power_log: bool = True   # パワースペクトルに log10 を取るか
    use_power_norm: bool = True  # パワースペクトルを (x - mean) / std で標準化するか
    log_eps: float = 1e-12       # log10 を取る際の ε


# ===================== 基本ユーティリティ =====================

def list_images(root: str | Path) -> List[Path]:
    """
    指定ディレクトリ以下の SEM 画像ファイルを再帰的に列挙する。

    Parameters
    ----------
    root : str | Path
        探索するルートディレクトリ

    Returns
    -------
    List[Path]
        画像ファイルへのパスのリスト（ソート済み）
    """
    root = Path(root)
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(
            f"{root} 以下に画像ファイルが見つかりません "
            f"(対応拡張子: {sorted(IMG_EXTS)})"
        )
    return files


# ===================== 画像読み込み・前処理 =====================

def load_sem_image(path: Path, cfg: SemFftConfig) -> np.ndarray:
    """
    単一の SEM 画像に対して基本的な前処理を行う。

    処理内容：
      1) 画像ファイルを読み込み、グレースケール (L) へ変換
      2) 指定サイズ (cfg.size x cfg.size) へリサイズ
      3) （オプション）3x3 メディアンフィルタによるノイズ除去
      4) （オプション）CLAHE による局所コントラスト強調

    Parameters
    ----------
    path : Path
        入力画像ファイルのパス
    cfg : SemFftConfig
        前処理パラメータ

    Returns
    -------
    np.ndarray
        前処理後の 2D グレースケール画像 (float32)
    """
    # 1) グレースケールで読み込み
    img = Image.open(path).convert("L")

    # 2) リサイズ（単純に size x size に揃える）
    if img.size != (cfg.size, cfg.size):
        img = img.resize((cfg.size, cfg.size), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32)

    # 3) メディアンフィルタ（塩胡椒ノイズに有効）
    if cfg.use_median:
        arr = median(arr, footprint=square(3)).astype(np.float32)

    # 4) CLAHE（局所コントラスト強調：必要な場合のみ使用）
    if cfg.use_clahe:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(
                "CLAHE を利用するには opencv-python が必要です。\n"
                "  pip install opencv-python\n"
                "を実行してください。"
            ) from e
        arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr_u8).astype(np.float32)

    return arr


def standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    配列 x を (x - mean) / std で標準化するヘルパー関数。

    Parameters
    ----------
    x : np.ndarray
        入力配列
    eps : float
        分散が極端に小さい場合のしきい値（ゼロ割防止）

    Returns
    -------
    np.ndarray
        標準化後の配列
    """
    mean = float(x.mean())
    std = float(x.std())
    if std < eps:
        std = eps
    return (x - mean) / std


def apply_hann_window(img: np.ndarray) -> np.ndarray:
    """
    2D Hann 窓（ハニング窓）を適用する。

    Notes
    -----
    - 画像の端での急激な値の変化によるスペクトルリーク
      （不要な高周波成分の混入）を抑制する効果がある。
    """
    h, w = img.shape
    wy = np.hanning(h)
    wx = np.hanning(w)
    window = np.outer(wy, wx).astype(np.float32)
    return img * window


# ===================== FFT / パワースペクトル =====================

def fft_power2d(img: np.ndarray, cfg: SemFftConfig) -> np.ndarray:
    """
    グレースケール画像から 2D パワースペクトルを計算する。

    処理フロー：
      1) （オプション）画素強度の標準化 (use_intensity_norm)
      2) （オプション）2D Hann 窓の適用 (use_window)
      3) 2D FFT の計算
      4) fftshift によりゼロ周波数を中心に移動
      5) P_raw = |F|^2
      6) （オプション）P_log = log10(P_raw + eps) (use_power_log)
      7) （オプション）P_std = (P_log - mean) / std (use_power_norm)

    Parameters
    ----------
    img : np.ndarray
        前処理済み 2D グレースケール画像
    cfg : SemFftConfig
        パイプライン設定

    Returns
    -------
    np.ndarray
        2D パワースペクトル（log・標準化などを適用後, float32）
    """
    arr = img.astype(np.float32)

    # 1) 強度標準化（任意）
    if cfg.use_intensity_norm:
        arr = standardize(arr)

    # 2) Hann 窓（任意）
    if cfg.use_window:
        arr = apply_hann_window(arr)

    # 3) 2D FFT
    F = np.fft.fft2(arr)
    # 4) ゼロ周波数成分を中心に移動
    F_shift = np.fft.fftshift(F)
    # 5) 生のパワースペクトル
    P = np.abs(F_shift) ** 2

    # 6) log10 変換（任意）
    if cfg.use_power_log:
        P = np.log10(P + cfg.log_eps)

    # 7) パワースペクトルの標準化（任意）
    if cfg.use_power_norm:
        P = standardize(P)

    return P.astype(np.float32)


def fft_power2d_from_path(path: Path, cfg: SemFftConfig) -> np.ndarray:
    """
    ファイルパスを受け取り：
      読み込み → 前処理 → FFT → 2D パワースペクトル
    までを一気に行うユーティリティ関数。
    """
    img = load_sem_image(path, cfg)
    P = fft_power2d(img, cfg)
    return P


# ===================== 2D パワースペクトルの PNG 保存 =====================

def save_power2d_png(P: np.ndarray, out_png: Path):
    """
    2D パワースペクトルを 0〜255 の範囲にスケーリングして PNG として保存する。

    Notes
    -----
    - ここでは log 変換や標準化は行わず、
      入力 P の値をそのまま min-max 正規化して表示用に変換する。
    """
    P = P.astype(np.float32)
    vmin, vmax = float(P.min()), float(P.max())
    if vmax - vmin < 1e-6:
        img_u8 = np.zeros_like(P, dtype=np.uint8)
    else:
        norm = (P - vmin) / (vmax - vmin)
        img_u8 = (norm * 255).astype(np.uint8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(out_png)


# ===================== ディレクトリ単位のパイプライン =====================

def process_directory(
    data_dir: str | Path,
    out_dir: str | Path,
    cfg: SemFftConfig
) -> List[Dict]:
    """
    指定ディレクトリ以下の全ての SEM 画像に対して
    「読み込み → 前処理 → FFT → 2D パワースペクトル → 保存」
    を一括で実行するパイプライン関数。

    Parameters
    ----------
    data_dir : str | Path
        入力 SEM 画像ディレクトリ
    out_dir : str | Path
        出力ルートディレクトリ
    cfg : SemFftConfig
        パイプライン設定

    Returns
    -------
    List[Dict]
        各画像ごとの情報をまとめた dict のリスト。
        例：
          {
             "file": 元画像パス,
             "power2d_npy": パワースペクトル .npy のパス,
             "power2d_png": パワースペクトル .png のパス
          }
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_dir = out_dir / "power2d_npy"
    png_dir = out_dir / "power2d_png"
    npy_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(data_dir)
    print(f"[INFO] {data_dir} から {len(files)} 枚の画像を検出しました。")

    records: List[Dict] = []

    for p in tqdm(files, desc="SEM → 2D FFT power"):
        P = fft_power2d_from_path(p, cfg)

        stem = p.stem
        npy_path = npy_dir / f"{stem}_power2d.npy"
        png_path = png_dir / f"{stem}_power2d.png"

        np.save(npy_path, P)
        save_power2d_png(P, png_path)

        records.append({
            "file": str(p),
            "power2d_npy": str(npy_path),
            "power2d_png": str(png_path),
        })

    return records


# ===================== CLI エントリポイント =====================

def main_cli() -> None:
    """
    コマンドラインから直接実行するためのエントリポイント。

    使用例：
      # 推奨設定（メディアン + 強度正規化 + Hann 窓 + log + 標準化）
      python sem_fft_pipeline.py \
        --data ./data/aachen_200/images_200 \
        --out  ./out_fft2d_aachen \
        --size 256

      # ほぼ生のスペクトル（前処理をオフ）
      python sem_fft_pipeline.py \
        --data ./data/aachen_200/images_200 \
        --out  ./out_fft2d_raw \
        --no-median --no-intensity-norm --no-window \
        --no-power-log --no-power-norm
    """
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser(
        description="SEM 画像 → 2D FFT パワースペクトル パイプライン"
    )
    ap.add_argument("--data", required=True,
                    help="入力 SEM 画像ディレクトリ")
    ap.add_argument("--out", required=True,
                    help="出力ルートディレクトリ")
    ap.add_argument("--size", type=int, default=256,
                    help="リサイズ後の画像サイズ（default: 256）")

    ap.add_argument("--no-median", action="store_true",
                    help="3x3 メディアンフィルタによるノイズ除去を無効化")
    ap.add_argument("--clahe", action="store_true",
                    help="CLAHE による局所コントラスト強調を有効化")
    ap.add_argument("--no-intensity-norm", action="store_true",
                    help="入力画像の強度標準化 (x-mean)/std を無効化")
    ap.add_argument("--no-window", action="store_true",
                    help="2D Hann 窓の適用を無効化")
    ap.add_argument("--no-power-log", action="store_true",
                    help="パワースペクトルに log10 を取る処理を無効化")
    ap.add_argument("--no-power-norm", action="store_true",
                    help="パワースペクトルの標準化 (x-mean)/std を無効化")

    args = ap.parse_args()

    cfg = SemFftConfig(
        size=args.size,
        use_median=not args.no_median,
        use_clahe=args.clahe,
        use_intensity_norm=not args.no_intensity_norm,
        use_window=not args.no_window,
        use_power_log=not args.no_power_log,
        use_power_norm=not args.no_power_norm,
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] 設定:", cfg)
    print(f"[INFO] 入力: {args.data}")
    print(f"[INFO] 出力: {out_root}")

    records = process_directory(args.data, out_root, cfg)

    # index.csv を書き出し
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(out_root / "index.csv", index=False, encoding="utf-8")

    print("[DONE] 2D パワースペクトル処理が完了しました。")
    print("  npy ディレクトリ :", (out_root / 'power2d_npy').resolve())
    print("  png ディレクトリ :", (out_root / 'power2d_png').resolve())
    print("  index.csv        :", (out_root / 'index.csv').resolve())


if __name__ == "__main__":
    main_cli()