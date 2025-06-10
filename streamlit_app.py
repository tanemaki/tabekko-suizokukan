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
    st.title("🐂 たべっこ水族館のキャラ当てアプリ")
    st.write(
        """
        「難しいことは量子アニーラーにやらせよう」を合言葉に、このアプリは開発されました。
        このアプリを使うことで、長年人類を悩ませてきた「たべっ子水族館」のキャラ当てが驚くほど簡単に。
        用意するのは、5cm四方の正方形が描かれた白いコピー用紙と「たべっこ水族館」のビスケットだけ。
        黒のボールペンで描いた正方形の中にビスケットを置いてね。
        下の図にあるキャラと近い角度で置くと正答率がぐんと上がるよ。
        是非あなたもお試しあれ。
        """
    )

    columns = st.columns(2)
    columns[0].image("./data/tabekko_table.jpg", caption="たべっこ水族館のキャラ一覧表")
    columns[1].image("./data/03_05.jpg", caption="置き方の例（正解：あしか）")

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

    if image_preprocessor.standardized_image is None:
        st.warning("この画像だとうまく読み取れなかったよ。別の画像を入れてみてね")
        visualize_image_processor(image_preprocessor)

        st.stop()

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

    dataset_visualizer = DatasetVisualizer(dataset, tabekko_df)

    top_class_ids = estimated_result["sorted_class_ids"][:5]

    with st.expander("上位ランキングされた動物を覚えるときに使ったデータ"):
        for selected_class_id in top_class_ids:
            dataset_visualizer.visualize_three_images(class_id=selected_class_id)

    with st.expander("画像の前処理"):
        visualize_image_processor(image_preprocessor)

    with st.expander("どう考えたのか見てみる"):
        visualize_result(estimated_result, estimator)

    with st.expander("モデルの頭の中（QUBO行列）を見てみる"):
        columns = st.columns(2)
        qubo = estimator.qubo
        figure, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(qubo, cmap="bwr", clim=(-2, 2))
        columns[0].pyplot(figure)

    with st.expander("覚えた形をみてみよう"):
        selected_class_id = st.selectbox(
            "select data",
            options=tabekko_df.index,
            format_func=lambda i: f"{i}: {tabekko_df.loc[i]['japanese_name']}",
        )

        dataset_visualizer.visualize_three_images(class_id=selected_class_id)


class DatasetVisualizer:
    def __init__(self, dataset, tabekko_df):
        self.dataset = dataset
        self.images = dataset["images"]
        self.labels = dataset["labels"]
        self.unique_class_ids = np.unique(self.labels)
        self.files = dataset["files"]

        self.tabekko_df = tabekko_df

    def visualize_three_images(self, class_id):
        figure, axs = plt.subplots(1, 3, figsize=(4, 1))

        indices = np.where(self.labels == class_id)[0]
        selected_indices = indices[:3]

        selected_images = self.images[selected_indices]
        selected_labels = self.labels[selected_indices]

        figure, axs = plt.subplots(1, 3, figsize=(12, 2))

        japanese_name = self.tabekko_df.loc[class_id]["japanese_name"]

        ax = axs[0]
        ax.imshow(selected_images[0], cmap="gray")
        ax.set_title(f"Class ID: {selected_labels[0]} ({japanese_name})")
        ax.axis("off")
        ax = axs[1]
        ax.imshow(selected_images[1], cmap="gray")
        ax.set_title(f"Class ID: {selected_labels[1]} ({japanese_name})")
        ax.axis("off")
        ax = axs[2]
        ax.imshow(selected_images[2], cmap="gray")
        ax.set_title(f"Class ID: {selected_labels[2]} ({japanese_name})")
        ax.axis("off")

        st.pyplot(figure)


def visualize_result(estimated_result, estimator):
    figure, axs = plt.subplots(1, 3, figsize=(22, 6))

    ax = axs[0]
    ax.imshow(estimated_result["binary_vector"].reshape(28, 28))
    ax.axis("off")

    estimated_class_id = estimated_result["estimated_class_id"]
    estimated_class_name = estimated_result["estimated_class_name"]

    energies = estimated_result["energies"]
    C = estimator.C

    ax = axs[1]
    ax.bar(np.arange(C), energies)
    ax.axhline(np.min(energies), color="r", linestyle="--")
    ax.set_title(f"Lowest energy label: {estimated_class_id} ({estimated_class_name})")
    ax.set_ylabel("Energy")
    ax.set_xticks(np.arange(C))
    ax.set_xticklabels(estimator.xtick_labels, rotation=270)
    max_energy = np.max(energies)
    min_energy = np.min(energies)
    energy_delta = max_energy - min_energy
    ax.set_ylim(min_energy - 0.1 * energy_delta, max_energy + 0.1 * energy_delta)

    probabilities = estimated_result["probabilities"]

    highest_probability_class_index = np.argmax(probabilities)
    highest_probability_class_id = estimator.unique_class_ids[
        highest_probability_class_index
    ]
    ax = axs[2]
    ax.bar(np.arange(len(probabilities)), probabilities)
    ax.set_title(
        f"Highest probability label: {highest_probability_class_id} (beta: {estimator.beta})"
    )
    ax.set_ylabel("Probability")
    ax.set_xticks(np.arange(C))
    ax.set_xticklabels(estimator.xtick_labels, rotation=270)
    ax.axhline(np.max(probabilities), color="r", linestyle="--")

    st.pyplot(figure)


def get_class_index(class_id, unique_class_ids):
    return int(np.where(unique_class_ids == class_id)[0][0])


if __name__ == "__main__":
    render_page()
