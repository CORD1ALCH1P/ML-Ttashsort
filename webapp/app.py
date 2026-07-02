"""Streamlit web app: upload a photo, classify the waste type, get a sorting recommendation."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import load_checkpoint, available_architectures  # noqa: E402
from waste_classes import CLASS_INFO  # noqa: E402
from inference import predict  # noqa: E402

REPORTS_DIR = ROOT / "reports"

st.set_page_config(page_title="Классификация мусора", page_icon="♻️", layout="centered")
st.title("♻️ Классификация мусора для сортировки отходов")
st.caption(
    "Загрузите фото отхода — система определит материал (бумага, пластик, стекло, "
    "металл, картон или прочий мусор), покажет вероятность и подскажет, в какой "
    "контейнер его сортировать."
)


@st.cache_resource
def get_model(arch: str):
    return load_checkpoint(arch)


@st.cache_data
def get_comparison_table():
    path = REPORTS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def get_best_arch():
    path = REPORTS_DIR / "best_model.json"
    if path.exists():
        return json.loads(path.read_text())["arch"]
    return None


archs = available_architectures()
if not archs:
    st.error("Обученные модели не найдены в папке `models/`. Сначала запустите обучение (src/train.py).")
    st.stop()

best_arch = get_best_arch()
default_idx = archs.index(best_arch) if best_arch in archs else 0

with st.sidebar:
    st.header("Настройки")
    arch = st.selectbox(
        "Модель (архитектура)",
        options=archs,
        index=default_idx,
        help="Можно сравнить, как разные архитектуры классифицируют одно и то же фото.",
    )
    if arch == best_arch:
        st.success("Рекомендованная лучшая модель по итогам сравнения")

    table = get_comparison_table()
    if table is not None:
        st.subheader("Сравнение моделей")
        st.dataframe(
            table.set_index("arch")[["accuracy", "macro_f1", "latency_ms_per_image", "model_size_mb"]]
            .round(3),
            use_container_width=True,
        )

model, classes, img_size = get_model(arch)

uploaded = st.file_uploader("Загрузите фото отхода", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded is not None:
    image = Image.open(uploaded)
    with col1:
        st.image(image, caption="Загруженное фото", use_container_width=True)

    probs, latency_ms = predict(model, classes, img_size, image)
    sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_class, top_prob = sorted_probs[0]
    info = CLASS_INFO[top_class]

    with col2:
        st.subheader(f"Класс: {info['ru_name']}")
        st.metric("Уверенность модели", f"{top_prob * 100:.1f}%")
        st.write(f"**Куда сортировать:** {info['bin']}")
        st.info(info["recommendation"])
        st.caption(f"Модель: {arch} · время инференса: {latency_ms:.1f} мс (CPU)")

    st.subheader("Вероятности по всем классам")
    df = pd.DataFrame(
        [{"Класс": CLASS_INFO[c]["ru_name"], "Вероятность": p} for c, p in sorted_probs]
    )
    st.bar_chart(df.set_index("Класс"))

    with st.expander("Сырые вероятности (JSON)"):
        st.json(probs)
else:
    st.info("Загрузите изображение, чтобы получить классификацию.")

st.divider()
st.caption(
    "Прототип системы сортировки отходов. Датасет: TrashNet (6 классов, ~2500 изображений). "
    "Модели: transfer learning на предобученных ImageNet-архитектурах, дообученных для CPU."
)
