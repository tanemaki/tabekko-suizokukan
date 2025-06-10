import streamlit as st
import numpy as np
import cv2

from tabekko.preprocessor import ImagePreProcessor

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

    st.image("./data/tabekko_table.jpg", caption="たべっこ水族館のキャラ一覧表", width=600)

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

    visualize_image_processor(image_preprocessor)

if __name__ == "__main__":
    render_page()