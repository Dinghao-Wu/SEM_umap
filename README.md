# SEM FFT パイプライン

SEM 画像を **2D パワースペクトル（|FFT|²）** に変換し，  
その後の次元削減・クラスタリング（UMAP / PCA / HDBSCAN など）に利用するための  
前処理パイプラインです。

- `sem_fft_pipeline.py`  
  - SEM 画像 → グレースケール → リサイズ → 前処理  
    → 2D FFT → 2D パワースペクトル
  - 結果を `.npy`（数値データ）と `.png`（可視化画像）で保存します。
- `sem_fft_hdbscan.py`（任意）  
  - `*_power2d.npy` を特徴量として読み込み，  
    PCA / UMAP + HDBSCAN によるクラスタリングを行います。

Wavebase に代表される「画像をパワースペクトルに変換して波形データとして扱う」という  
考え方を，自前の SEM データに適用することを目的としています。

---

## 背景とコンセプト

材料科学における SEM 画像には，粒径，粒子間距離，配向性（ラメラ構造，ストライプ）などの  
**周期性・スケール情報**が含まれます。

ただし，画像のままピクセル空間で扱うと，

- 視野の平行移動や切り出し位置に敏感
- 明るさやコントラストの違いが直接影響する
- 高次元（画素数）のため，そのままでは扱いにくい

といった問題があります。

そこで本パイプラインでは，SEM 画像を **2D の波形データ**とみなし，

1. 2D フーリエ変換（FFT）を行い
2. 2D パワースペクトル \(|F|^2\) を計算し
3. 必要に応じて log 変換・標準化を行う

ことで，「どの方向・どの空間周波数の構造がどれくらい強いか」という  
**統計的な構造指紋**を得ます。  
この指紋をベクトルとして保存しておくことで，後段の UMAP / PCA / HDBSCAN などの  
機械学習・クラスタリングへスムーズに接続できます。

---

## 動作環境

- Python 3.9 以上（3.12 で動作確認）
- 必要なパッケージ

```bash
pip install numpy pillow tqdm scikit-image pandas matplotlib
# クラスタリングまで行う場合
pip install scikit-learn umap-learn hdbscan

1. SEM → 2D パワースペクトル（sem_fft_pipeline.py）
基本的な使い方
python sem_fft_pipeline.py \
  --data ./data/sem_images \
  --out  ./out_fft2d_sem \
  --size 256


--data : 入力 SEM 画像ディレクトリ（png / jpg / tif 等）

--out : 出力ルートディレクトリ

--size : リサイズ後の画像サイズ（size × size，デフォルト 256）

前処理オプション

sem_fft_pipeline.py では，FFT 前にいくつかの前処理をオン／オフできます。

--no-median

3×3 メディアンフィルタによるノイズ除去を無効化

デフォルト：有効（塩胡椒ノイズ抑制のため）

--clahe

CLAHE（局所コントラスト強調）を有効化

デフォルト：無効（人工的な高周波を増やさないため）

--no-intensity-norm

入力画像の強度標準化 (x - mean) / std を無効化

デフォルト：有効（明るさの違いを吸収するため）

--no-window

2D Hann 窓の適用を無効化

デフォルト：有効（端の切断によるスペクトルリークを抑制）

--no-power-log

パワースペクトルに対する log10(|F|^2 + eps) 変換を無効化

デフォルト：有効（ダイナミックレンジ圧縮のため）

--no-power-norm

パワースペクトルの標準化 (x - mean) / std を無効化

デフォルト：有効（サンプル間のスケールを揃えるため）

出力

--out ./out_fft2d_sem の場合，以下のような構成が生成されます。

out_fft2d_sem/
  ├── power2d_npy/
  │     ├── IMG_00001_power2d.npy
  │     ├── IMG_00002_power2d.npy
  │     └── ...
  ├── power2d_png/
  │     ├── IMG_00001_power2d.png
  │     ├── IMG_00002_power2d.png
  │     └── ...
  └── index.csv


power2d_npy/

各 SEM 画像の 2D パワースペクトル（numpy.ndarray, float32, shape ≒ [size, size]）

power2d_png/

パワースペクトルを log スケール＋min-max 正規化した可視化画像

index.csv

file, power2d_npy, power2d_png の対応表

.npy の簡単な確認方法
import numpy as np
import matplotlib.pyplot as plt

P = np.load("out_fft2d_sem/power2d_npy/IMG_00001_power2d.npy")
print("shape:", P.shape)
print("min, max:", P.min(), P.max())

plt.imshow(P, cmap="inferno")
plt.colorbar()
plt.title("2D Power Spectrum")
plt.show()