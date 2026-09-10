# app.py — Updated to download models from Hugging Face
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Tourism Analytics", page_icon="🌍", layout="wide")


# ============================================================
# MODEL DOWNLOAD FROM HUGGING FACE
# ============================================================
MODEL_URL = (
    "https://huggingface.co/M-Prasad/TOURIST-PREDICT/resolve/main/tourism_models.pkl"
)
MODEL_PATH = Path("tourism_models.pkl")


@st.cache_resource(show_spinner=False)
def download_model_if_needed(url: str, dest: Path) -> bool:
    """Download the model from Hugging Face if not already present."""
    # If file exists AND is complete (not partial), skip download
    if dest.exists() and dest.stat().st_size > 1_000_000:  # > 1 MB sanity check
        st.success(f"✅ Model already available ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    try:
        with st.spinner("📥 Downloading model from Hugging Face (~89 MB)..."):
            # Stream download to handle large files
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                chunk_size = 1024 * 1024  # 1 MB chunks

                progress_bar = st.progress(0.0)
                downloaded = 0

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                progress_bar.progress(downloaded / total)

                progress_bar.empty()
        st.success(f"✅ Model downloaded ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to download model: {e}")
        return False


# Download model on first run
if not download_model_if_needed(MODEL_URL, MODEL_PATH):
    st.stop()


@st.cache_resource(show_spinner="Loading model into memory...")
def load_models():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    master = pd.read_csv("master_dataset_clean.csv")
    users = pd.read_csv("user_features.csv")
    attractions = pd.read_csv("attraction_features.csv")
    return master, users, attractions


# ---- Load everything ----
artifacts = load_models()
master_df, user_features, attraction_features = load_data()

st.title("🌍 Tourism Analytics & Personalized Recommendations")
st.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Analytics", "⭐ Predict Rating",
    "🎯 Predict Visit Mode", "🎁 Recommendations"
])


# ============================================================
# 1. ANALYTICS
# ============================================================
if page == "📊 Analytics":
    st.header("Tourism Analytics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(master_df):,}")
    c2.metric("Users", f"{master_df['UserId'].nunique():,}")
    c3.metric("Attractions", f"{master_df['AttractionId'].nunique():,}")
    c4.metric("Avg Rating", f"{master_df['Rating'].mean():.2f}")

    st.subheader("Rating Distribution")
    st.bar_chart(master_df['Rating'].value_counts().sort_index())

    st.subheader("Visit Mode Distribution")
    st.bar_chart(master_df['VisitModeName'].value_counts())

    st.subheader("Top 10 Attractions")
    top = master_df.groupby('Attraction').size().sort_values(ascending=False).head(10)
    st.bar_chart(top)


# ============================================================
# 2. PREDICT RATING
# ============================================================
elif page == "⭐ Predict Rating":
    st.header("Predict Attraction Rating")

    col1, col2 = st.columns(2)

    with col1:
        year = st.slider("Visit Year", 2015, 2025, 2024)
        month = st.slider("Visit Month", 1, 12, 6)
        mode = st.selectbox(
            "Visit Mode", [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: 'Business', 2: 'Couples', 3: 'Family',
                4: 'Friends', 5: 'Solo'
            }[x]
        )

    with col2:
        unique_attractions = sorted(master_df['Attraction'].dropna().unique())
        attraction_name = st.selectbox("Attraction", unique_attractions)
        season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])

    if st.button("Predict Rating", type="primary"):
        # ---- Lookup real feature values ----
        row = master_df[master_df['Attraction'] == attraction_name].iloc[0]

        attraction_type_str = row['AttractionType']
        attraction_avg = float(row['AttractionAvgRating'])
        attraction_visits = float(row['AttractionTotalVisits'])

        user_avg_rating = float(master_df['UserAvgRating'].mean())
        user_visit_count = float(master_df['UserVisitCount'].median())
        user_unique_attr = float(master_df['UserUniqueAttractions'].median())

        # ---- Encode using SAVED encoders ----
        try:
            atype_enc = int(artifacts['le_type'].transform([attraction_type_str])[0])
        except Exception:
            atype_enc = 0

        mode_name = {1: 'Business', 2: 'Couples', 3: 'Family',
                     4: 'Friends', 5: 'Solo'}[mode]
        try:
            mode_enc = int(artifacts['le_mode'].transform([mode_name])[0])
        except Exception:
            mode_enc = 0

        try:
            season_enc = int(artifacts['le_season'].transform([season])[0])
        except Exception:
            season_enc = 0

        # ---- Build input row ----
        input_d = {
            'VisitYear': year,
            'VisitMonth': month,
            'VisitMode': mode,
            'AttractionType_enc': atype_enc,
            'VisitModeName_enc': mode_enc,
            'Season_enc': season_enc,
            'UserAvgRating': user_avg_rating,
            'UserVisitCount': user_visit_count,
            'UserUniqueAttractions': user_unique_attr,
            'AttractionAvgRating': attraction_avg,
            'AttractionTotalVisits': attraction_visits,
        }
        X = pd.DataFrame([input_d])[artifacts['regression_features']]

        # ---- Scale using SAVED scaler ----
        X_scaled = artifacts['reg_scaler'].transform(X)

        # ---- Predict ----
        pred = float(np.clip(
            artifacts['regression_model'].predict(X_scaled)[0], 1, 5
        ))

        st.success(f"### ⭐ Predicted Rating: {pred:.2f} / 5.0")

        c1, c2, c3 = st.columns(3)
        c1.metric("Attraction Avg", f"{attraction_avg:.2f}")
        c2.metric("Total Visits", f"{int(attraction_visits)}")
        c3.metric("Model", artifacts.get('best_reg_name', 'Regressor'))

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            gauge={
                'axis': {'range': [1, 5]},
                'bar': {'color': "#1E88E5"},
                'steps': [
                    {'range': [1, 2], 'color': "#FFCDD2"},
                    {'range': [2, 3], 'color': "#FFE0B2"},
                    {'range': [3, 4], 'color': "#FFF9C4"},
                    {'range': [4, 5], 'color': "#C8E6C9"},
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 3. PREDICT VISIT MODE
# ============================================================
elif page == "🎯 Predict Visit Mode":
    st.header("Predict Visit Mode")

    col1, col2 = st.columns(2)

    with col1:
        year = st.slider("Year", 2015, 2025, 2024)
        month = st.slider("Month", 1, 12, 6)
        season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])

    with col2:
        attraction_name = st.selectbox(
            "Attraction",
            sorted(master_df['Attraction'].dropna().unique())[:500]
        )
        rating = st.slider("Expected Rating", 1.0, 5.0, 4.0)

    if st.button("Predict Visit Mode", type="primary"):
        row = master_df[master_df['Attraction'] == attraction_name].iloc[0]

        attraction_type_str = row['AttractionType']
        attraction_avg = float(row['AttractionAvgRating'])
        attraction_visits = float(row['AttractionTotalVisits'])

        user_avg_rating = float(master_df['UserAvgRating'].mean())
        user_visit_count = float(master_df['UserVisitCount'].median())
        user_unique_attr = float(master_df['UserUniqueAttractions'].median())

        try:
            atype_enc = int(artifacts['le_type'].transform([attraction_type_str])[0])
        except Exception:
            atype_enc = 0

        try:
            season_enc = int(artifacts['le_season'].transform([season])[0])
        except Exception:
            season_enc = 0

        input_d = {
            'VisitYear': year,
            'VisitMonth': month,
            'AttractionType_enc': atype_enc,
            'Season_enc': season_enc,
            'UserAvgRating': user_avg_rating,
            'UserVisitCount': user_visit_count,
            'UserUniqueAttractions': user_unique_attr,
            'AttractionAvgRating': attraction_avg,
            'AttractionTotalVisits': attraction_visits,
            'Rating': rating,
        }
        X = pd.DataFrame([input_d])[artifacts['classification_features']]
        X_scaled = artifacts['clf_scaler'].transform(X)

        pred = artifacts['classification_model'].predict(X_scaled)[0]
        proba = artifacts['classification_model'].predict_proba(X_scaled)[0]
        mode_name = artifacts['le_target'].inverse_transform([pred])[0]

        st.success(f"### 🎯 Predicted Visit Mode: **{mode_name}**")

        proba_df = pd.DataFrame({
            'Mode': artifacts['le_target'].classes_,
            'Probability': proba
        }).sort_values('Probability', ascending=True)

        fig = px.bar(proba_df, x='Probability', y='Mode', orientation='h',
                     color='Probability', color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=350,
                          title="Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 4. RECOMMENDATIONS (updated)
# ============================================================
elif page == "🎁 Recommendations":
    st.header("Personalized Recommendations")
    st.caption("Hybrid recommender combining collaborative, content-based, "
               "and popularity signals.")

    uid = st.selectbox(
        "Select User",
        sorted(user_features['UserId'].unique())[:500]
    )
    n = st.slider("Number of Recommendations", 5, 20, 10)

    if st.button("Get Recommendations", type="primary"):
        uim = artifacts['user_item_matrix']
        pop_scores = artifacts['popularity_scores']

        # ---------- Case 1: User exists in matrix ----------
        if uid in uim.index:
            idx = list(uim.index).index(uid)
            pred = artifacts['user_factors'][idx] @ artifacts['item_factors'].T

            rec = pd.DataFrame({
                'AttractionId': uim.columns,
                'RawScore': pred
            })

            visited = set(uim.loc[uid][uim.loc[uid] > 0].index)
            rec = rec[~rec['AttractionId'].isin(visited)]

            # Normalize CF scores to [0, 1]
            mn, mx = rec['RawScore'].min(), rec['RawScore'].max()
            if mx > mn:
                rec['CF_Score'] = (rec['RawScore'] - mn) / (mx - mn)
            else:
                rec['CF_Score'] = 0.5

            # Popularity (already 0-1)
            rec['Pop_Score'] = rec['AttractionId'].map(pop_scores).fillna(0)

            # Hybrid blend
            rec['Score'] = 0.7 * rec['CF_Score'] + 0.3 * rec['Pop_Score']

            rec = rec.sort_values('Score', ascending=False).head(n)

        # ---------- Case 2: Cold-start user ----------
        else:
            st.info("New user — showing popularity-based recommendations.")
            rec = attraction_features.sort_values(
                'AttractionTotalVisits', ascending=False
            ).head(n).copy()

            vmax = rec['AttractionTotalVisits'].max()
            rec['Score'] = rec['AttractionTotalVisits'] / vmax if vmax > 0 else 0
            rec = rec[['AttractionId', 'Score']]

        # ---------- Attach display info ----------
        rec = rec.merge(
            master_df[['AttractionId', 'Attraction', 'AttractionType']]
              .drop_duplicates('AttractionId'),
            on='AttractionId', how='left'
        )
        rec = rec[['AttractionId', 'Attraction', 'AttractionType', 'Score']]
        rec['Score'] = rec['Score'].round(4)

        st.subheader(f"Top {n} Recommendations for User {uid}")
        st.dataframe(rec, use_container_width=True)

        # Bar chart
        fig = px.bar(
            rec.sort_values('Score'),
            x='Score', y='Attraction', orientation='h',
            color='Score', color_continuous_scale='Viridis',
        )
        fig.update_layout(
            height=max(400, n * 30),
            yaxis={'categoryorder': 'total ascending'},
            title=f"Recommendation Scores for User {uid}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download
        st.download_button(
            "📥 Download CSV",
            rec.to_csv(index=False),
            f"recommendations_user_{uid}.csv",
            mime='text/csv'
        )