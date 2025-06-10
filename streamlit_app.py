import streamlit as st
import numpy as np
import cv2

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

    st.image(image, channels='BGR')

if __name__ == "__main__":
    render_page()