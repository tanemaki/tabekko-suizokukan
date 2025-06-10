import urllib.request
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from tabekko.estimator import TabekkoSuizokukanEstimator
from tabekko.preprocessor import ImagePreProcessor

# 保存先ディレクトリ（プロジェクト内の .fonts フォルダなど）
font_dir = Path(".fonts")
font_dir.mkdir(exist_ok=True)
font_path = font_dir / "NotoSansCJKjp-Regular.otf"


# フォントが存在しなければダウンロード
if not font_path.exists():
    print("Downloading Noto Sans CJK JP font...")
    url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
    urllib.request.urlretrieve(url, font_path)
    print("Font downloaded to:", font_path)

# フォントを読み込んで設定
jp_font = fm.FontProperties(fname=str(font_path))
fm.fontManager.addfont(str(font_path))
mpl.rcParams["font.family"] = jp_font.get_name()


def visualize_image_processor(image_preprocessor):
    # 画像中の正方形の枠線を認識し、角の4点の座標を読み取る処理を可視化
    image_columns = st.columns(3)
    image_columns[0].image(
        image_preprocessor.resized_original_image,
        channels="BGR",
        caption="撮影画像（640x480にリサイズ済み）",
    )
    image_columns[1].image(
        image_preprocessor.edges, channels="GRAY", caption="エッジ検出後"
    )
    # Display the image with detected squares
    image_columns[2].image(
        image_preprocessor.annotated_image, channels="BGR", caption="正方枠の検出"
    )

    if image_preprocessor.num_squares == 0:
        # 枠線が見つからなければストップ
        st.warning(
            "正方形の枠線がうまく見つけられませんでした😭　他の画像を入れてみてください🙇‍♂️"
        )
        st.stop()

    # 台形補正で切り出した正方画像の処理を可視化
    image_columns = st.columns(4)
    image_columns[0].image(
        image_preprocessor.square_image,
        channels="BGR",
        width=200,
        caption="検出領域の抽出",
    )
    image_columns[1].image(image_preprocessor.binary_image, width=200, caption="二値化")
    image_columns[2].image(
        image_preprocessor.noise_removed_image, width=200, caption="ノイズ除去"
    )
    image_columns[3].image(
        image_preprocessor.standardized_image,
        channels="GRAY",
        width=200,
        caption=f"{image_preprocessor.edge_length}x{image_preprocessor.edge_length}の画像",
    )


def render_page():
    # Show the page title and description.
    st.set_page_config(page_title="たべっこ水族館", page_icon="🐟", layout="wide")
    st.title("🐟たべっこ水族館のキャラ当てアプリ")
    st.write(
        """
        「難しいことは量子アニーラーにやらせよう」を合言葉に、このアプリは開発されました。
        このアプリを使うことで、長年人類を悩ませてきた「たべっ子水族館」のキャラ当てが驚くほど簡単に。
        用意するのは、5cm四方の正方形が描かれた白いコピー用紙と「たべっこ水族館」のビスケットだけ。
        黒のボールペンで描いた正方形の中に、ビスケットを置いてね。
        下の図にあるキャラと近い角度で置くと正答率がぐんと上がるよ。
        是非あなたもお試しあれ。
        """
    )

    st.image(
        "./data/tabekko_table.jpg", caption="たべっこ水族館のキャラ一覧表", width=600
    )

    uploaded_file = st.file_uploader(
        "キャラ当ての画像をアップロードしてください", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:
        st.warning("画像がアップロードされていません。")
        st.stop()

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # OpenCVで画像として読み込み（BGR形式）
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 画像を表示
    columns = st.columns(2)
    columns[0].image(image, channels="BGR", caption="アップロードした画像")

    edge_length = 28

    image_preprocessor = ImagePreProcessor(image, edge_length)

    tabekko_df = pd.read_csv("data/ginbis_tabekko_suizokukan.csv")
    tabekko_df = tabekko_df.set_index("id")

    dataset = np.load("data/250512a/dataset.npz")

    # 使用可能な動物IDのリスト
    unique_class_ids = np.unique(dataset["labels"])

    # 学習済みのQUBO行列等を読み込む
    loaded_result = np.load("data/250512a/result.npz")

    qubo = loaded_result["qubo"]

    estimator = TabekkoSuizokukanEstimator(
        qubo,
        unique_class_ids,
        loaded_result["nodes_per_class"],
        edge_length,
        tabekko_df,
    )

    visualize_image_processor(image_preprocessor)
    estimated_result = estimator.estimate(image_preprocessor.standardized_image)

    columns[1].info(f"これはたぶん「{estimated_result['estimated_class_name']}」だよ！")
    result_df = pd.DataFrame(
        {
            "名前": estimated_result["sorted_class_names"].values[:5],
            "確率": estimated_result["sorted_probabilities"][:5],
        }
    )

    result_df.set_index("名前", inplace=True)
    figure, ax = plt.subplots(figsize=(10, 5))
    result_df.plot(kind="barh", ax=ax)
    ax.set_title("確率の高い順に上位5つ")
    ax.set_xlabel("確率")
    ax.legend().remove()
    plt.gca().invert_yaxis()
    columns[1].pyplot(figure)


if __name__ == "__main__":
    render_page()
