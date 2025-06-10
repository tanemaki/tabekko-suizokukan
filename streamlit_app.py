import altair as alt
import pandas as pd
import streamlit as st

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


if __name__ == "__main__":
    render_page()