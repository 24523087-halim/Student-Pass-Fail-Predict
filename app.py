import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Klasifikasi Kelulusan Siswa",
    layout="wide",
)

st.title("Klasifikasi Kelulusan Siswa")
st.markdown("Project Akhir untuk memprediksi kelulusan dengan beberapa algoritma ML.")
st.markdown("---")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv("student_data.csv")


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame):
    """Encode fitur kategori dan buat label target lulus."""
    df_processed = df.copy()
    label_encoders = {}

    for col in df_processed.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        df_processed[col] = encoder.fit_transform(df_processed[col])
        label_encoders[col] = encoder

    df_processed["status_lulus"] = (df_processed["G3"] >= 10).astype(int)
    return df_processed, label_encoders


df_raw = load_data()
df_processed, label_encoders = preprocess_data(df_raw)

st.sidebar.header("Pengaturan Model")

mode_prediksi = st.sidebar.selectbox(
    "Mode Prediksi",
    ["Early Prediction (tanpa G1 & G2)", "Mid/End Semester (dengan G1 & G2)"],
    help="Pilih sesuai ketersediaan nilai G1 dan G2.",
)

algoritma = st.sidebar.selectbox(
    "Algoritma Machine Learning",
    ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes"],
)

st.sidebar.markdown("---")
if "Early" in mode_prediksi:
    st.sidebar.info(
        "Early Prediction: untuk siswa yang belum punya nilai G1 dan G2. "
        "Fokus pada demografi, perilaku, dan dukungan belajar."
    )
else:
    st.sidebar.success(
        "Mid/End Semester: Sudah mempunyai nilai G1 dan G2."
    )
st.sidebar.markdown("- Data train: 80%")
st.sidebar.markdown("- Data test: 20%")

st.subheader("Statistik Data Siswa")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Siswa", len(df_raw))
with col2:
    lulus = (df_raw["G3"] >= 10).sum()
    st.metric("Siswa Lulus", f"{lulus} ({lulus/len(df_raw)*100:.1f}%)")
with col3:
    tidak_lulus = (df_raw["G3"] < 10).sum()
    st.metric("Tidak Lulus", f"{tidak_lulus} ({tidak_lulus/len(df_raw)*100:.1f}%)")

st.markdown("---")

with st.expander("Lihat sample data"):
    st.dataframe(df_raw.head(20))
    st.caption(
        "Definisi label: lulus jika nilai akhir (G3) minimal 10, tidak lulus jika di bawahnya."
    )

base_features = [
    "studytime",
    "failures",
    "absences",
    "higher",
    "schoolsup",
    "famsup",
    "paid",
    "Medu",
    "Fedu",
    "goout",
    "Dalc",
    "Walc",
    "freetime",
    "internet",
    "age",
    "health",
]
full_features = ["G1", "G2"] + base_features

if "Early" in mode_prediksi:
    selected_features = base_features
    mode_label = "Early Prediction"
else:
    selected_features = full_features
    mode_label = "Mid/End Semester"

with st.expander(f"Fitur yang dipakai ({mode_label})"):
    if mode_label == "Mid/End Semester":
        st.markdown("- **Nilai:** `G1`, `G2`")
    st.markdown(
        "- **Akademik & dukungan:** `studytime`, `higher`, `schoolsup`, `famsup`, "
        "`paid`, `Medu`, `Fedu`, `internet`"
    )
    st.markdown(
        "- **Risiko:** `failures`, `absences`, `goout`, `Dalc`, `Walc`, `freetime`"
    )
    st.markdown("- **Tambahan:** `age`, `health`")

X = df_processed[selected_features]
y = df_processed["status_lulus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if algoritma == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, random_state=42, C=0.5, solver="lbfgs")
    model_name = "Logistic Regression"
elif algoritma == "Decision Tree":
    model = DecisionTreeClassifier(
        random_state=42, max_depth=4, min_samples_split=10, min_samples_leaf=5
    )
    model_name = "Decision Tree"
elif algoritma == "Random Forest":
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
    )
    model_name = "Random Forest"
else:
    model = GaussianNB(var_smoothing=1e-8)
    model_name = "Naive Bayes"

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

st.header(f"Hasil klasifikasi: {model_name}")

if mode_label == "Early Prediction":
    st.info(f"Mode: {mode_label} (tanpa G1 dan G2) dengan {len(selected_features)} fitur.")
else:
    st.success(f"Mode: {mode_label} dengan {len(selected_features)} fitur.")

st.subheader("Metrik evaluasi")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.metric("Accuracy", f"{accuracy*100:.2f}%")
with metric_col2:
    st.metric("Precision", f"{precision*100:.2f}%")
with metric_col3:
    st.metric("Recall", f"{recall*100:.2f}%")
with metric_col4:
    st.metric("F1-Score", f"{f1*100:.2f}%")

with st.expander("Penjelasan singkat metrik"):
    st.markdown(
        "- **Accuracy:** seberapa sering model menebak dengan benar dari semua kasus.\n"
        "- **Precision:** dari prediksi lulus, berapa yang benar-benar lulus (hindari salah anggap lulus).\n"
        "- **Recall:** dari semua siswa yang lulus, berapa yang berhasil ditemukan model.\n"
        "- **F1-Score:** rata-rata seimbang antara precision dan recall."
    )

st.markdown("---")

st.subheader("Visualisasi")
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.markdown("**Confusion Matrix**")
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Tidak Lulus", "Lulus"],
        yticklabels=["Tidak Lulus", "Lulus"],
        annot_kws={"size": 14},
    )
    ax1.set_xlabel("Prediksi", fontsize=12)
    ax1.set_ylabel("Aktual", fontsize=12)
    ax1.set_title("Confusion Matrix", fontsize=14)
    st.pyplot(fig1)
    st.caption(
        f"- TP={tp}: prediksi lulus dan benar.\n"
        f"- FP={fp}: diprediksi lulus padahal tidak.\n"
        f"- TN={tn}: prediksi tidak lulus dan benar.\n"
        f"- FN={fn}: diprediksi tidak lulus padahal lulus."
    )

with viz_col2:
    st.markdown("**Perbandingan metrik**")
    fig2, ax2 = plt.subplots(figsize=(6, 5.75))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    bars = ax2.bar(metrics, values, color=colors, edgecolor="black")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Persentase (%)", fontsize=12)
    ax2.set_title("Metrik evaluasi", fontsize=14)
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax2.axhline(y=80, color="red", linestyle="--", alpha=0.5, label="Target 80%")
    ax2.legend()
    st.pyplot(fig2)
    st.caption(
        "Semakin tinggi semakin baik."
    )

st.markdown("---")

st.subheader("Distribusi hasil prediksi")
dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    st.markdown("**Data aktual**")
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    actual_counts = pd.Series(y_test).value_counts()
    labels = ["Lulus" if i == 1 else "Tidak Lulus" for i in actual_counts.index]
    colors_pie = ["#2ecc71", "#e74c3c"]
    ax3.pie(
        actual_counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors_pie,
        explode=(0.05, 0.05),
        shadow=True,
        startangle=90,
    )
    ax3.set_title("Distribusi aktual", fontsize=14)
    st.pyplot(fig3)
    st.caption("Proporsi label asli di data test.")

with dist_col2:
    st.markdown("**Hasil prediksi**")
    fig4, ax4 = plt.subplots(figsize=(5, 5))
    pred_counts = pd.Series(y_pred).value_counts()
    labels = ["Lulus" if i == 1 else "Tidak Lulus" for i in pred_counts.index]
    colors_pie = ["#2ecc71", "#e74c3c"]
    ax4.pie(
        pred_counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors_pie,
        explode=(0.05, 0.05),
        shadow=True,
        startangle=90,
    )
    ax4.set_title("Distribusi prediksi", fontsize=14)
    st.pyplot(fig4)
    st.caption("Perbandingan hasil tebakkan model dengan data aktual di samping.")

st.markdown("---")

if hasattr(model, "feature_importances_"):
    st.subheader("Faktor paling berpengaruh")
    importance_df = (
        pd.DataFrame({"Fitur": selected_features, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=True)
        .tail(10)
    )
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.barh(importance_df["Fitur"], importance_df["Importance"], color="steelblue")
    ax5.set_xlabel("Tingkat kepentingan", fontsize=12)
    ax5.set_title("Top 10 fitur", fontsize=14)
    st.pyplot(fig5)
    st.caption(
        "Bar lebih panjang berarti fitur lebih berperan dalam keputusan model pada training data."
    )
elif algoritma == "Logistic Regression":
    st.subheader("Koefisien model")
    coef_df = (
        pd.DataFrame({"Fitur": selected_features, "Koefisien": model.coef_[0]})
        .sort_values("Koefisien", key=abs, ascending=True)
        .tail(10)
    )
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    colors_coef = ["green" if val > 0 else "red" for val in coef_df["Koefisien"]]
    ax5.barh(coef_df["Fitur"], coef_df["Koefisien"], color=colors_coef)
    ax5.set_xlabel("Koefisien", fontsize=12)
    ax5.set_title("Top 10 fitur (Logistic Regression)", fontsize=14)
    ax5.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    st.pyplot(fig5)
    st.caption(
        "Koefisien positif meningkatkan peluang lulus; negatif menurunkan."
    )

st.markdown("---")
st.subheader("Kesimpulan singkat")

if accuracy >= 0.8:
    label_perf = "sangat baik"
elif accuracy >= 0.7:
    label_perf = "baik"
elif accuracy >= 0.6:
    label_perf = "cukup"
else:
    label_perf = "perlu ditingkatkan"

st.success(
    f"Model {model_name} {label_perf}: "
    f"accuracy {accuracy*100:.2f}%, precision {precision*100:.2f}%, "
    f"recall {recall*100:.2f}%, F1 {f1*100:.2f}%."
)

st.caption(
    "Catatan: model dilatih dengan pembagian data 80% train set & 20% test set dan fitur dipilih "
    "berdasarkan mode prediksi yang aktif."
)
