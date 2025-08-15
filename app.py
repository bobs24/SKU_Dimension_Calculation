from supabase import create_client, Client
import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
from datetime import datetime
import io
import os

from dotenv import load_dotenv
load_dotenv()

# Supabase credentials from .env
SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]
SUPABASE_BUCKET = st.secrets["supabase"]["SUPABASE_BUCKET"]

# Create a Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set page config
st.set_page_config(page_title="Fashion Packaging Prediction", layout="centered")

# Hide Streamlit’s default menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load models and preprocessor silently
model_dir = os.path.join(os.path.dirname(__file__), "model_folder")
preprocessor = pickle.load(open(os.path.join(model_dir, 'preprocessor.pkl'), 'rb'))
model_part1 = lgb.Booster(model_file=os.path.join(model_dir, 'model_0.txt'))
model_part2 = lgb.Booster(model_file=os.path.join(model_dir, 'model_1.txt'))
model_part3 = lgb.Booster(model_file=os.path.join(model_dir, 'model_2.txt'))
model_part4 = lgb.Booster(model_file=os.path.join(model_dir, 'model_3.txt'))

# App title and description
st.markdown("<h2 style='text-align: center;'>📦 Apparel Dimension & Weight Predictor</h2>", unsafe_allow_html=True)

# Download Excel Template
@st.cache_data
def generate_template():
    template_df = pd.DataFrame(columns=[
        'SKU Universal', 'Product Name', 'Brand',
        'Category', 'Subcategory', 'Gender', 'Size'
    ])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Template')
    return output.getvalue()

st.markdown("### 📥 Download Excel Template")
st.markdown("Click below to download the Excel template. Fill it with your product data and upload it back for predictions.")

st.download_button(
    label="Download Excel Template",
    data=generate_template(),
    file_name="prediction_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


st.markdown("### 📤 Upload Excel Template")
st.markdown(
    "Upload your Excel file to predict each product's  **Panjang**, **Lebar**, **Tinggi**, and **Berat**",
    unsafe_allow_html=True
)

# Upload file
# uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])
uploaded_file = st.file_uploader("Drop your file below", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error("⚠️ Error reading the Excel file. Please check the format.")
        st.stop()

    # Required columns
    required_columns = ['SKU Universal', 'Product Name', 'Brand', 'Category', 'Subcategory', 'Gender', 'Size']
    df[required_columns] = df[required_columns].astype(str)

    if not all(col in df.columns for col in required_columns):
        st.error("❌ Uploaded file is missing one or more required columns.")
        st.markdown(f"**Required columns:** {', '.join(required_columns)}")
        st.stop()

    # Transform data
    try:
        df_transformed = preprocessor.transform(df[required_columns])
    except Exception:
        st.error("⚠️ Failed to preprocess the data.")
        st.stop()

    # Predict
    custom_round_set = np.array([1000, 2000, 2500, 3000, 4500, 6000])

    try:
        predictions = []
        for i, model in enumerate([model_part1, model_part2, model_part3, model_part4]):
            y_pred = model.predict(df_transformed)
            if i == 3:  # Berat
                y_pred = np.expm1(y_pred)
                final_pred = []
                for val in y_pred:
                    diffs = np.abs(custom_round_set - val)
                    if np.min(diffs) < 500:
                        closest = custom_round_set[np.argmin(diffs)]
                        final_pred.append(closest)
                    else:
                        final_pred.append(np.round(val / 500) * 500)
                y_pred = np.array(final_pred)
            else:
                y_pred = np.ceil(y_pred)  # Round up
            predictions.append(y_pred)

        df['Panjang'] = predictions[0]
        df['Lebar']   = predictions[1]
        df['Tinggi']  = predictions[2]
        df['Berat']   = predictions[3]
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.stop()


    # Show results
    st.success("✅ Prediction completed!")
    with st.expander("📊 View Prediction Results"):
        st.dataframe(df[['SKU Universal', 'Product Name', 'Brand', 'Category', 'Subcategory', 'Gender', 'Size', 'Panjang', 'Lebar', 'Tinggi', 'Berat']])

    # Upload the Excel result to Supabase
    def upload_to_supabase(file_name, file_data):
        # Upload the file to the Supabase storage bucket
        base_name, ext = os.path.splitext(file_name)
        timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
        file_name = f"{base_name}_{timestamp}.xlsx"
        response = supabase.storage.from_(SUPABASE_BUCKET).upload(file_name, file_data)

    # Option to download and upload to Supabase
    @st.cache_data
    def convert_df(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        processed_data = output.getvalue()
        return processed_data

    # Upload the result to Supabase after conversion
    if uploaded_file is not None:
        file_name = "predicted_dimensions.xlsx"
        file_data = convert_df(df)

        # Upload the result to Supabase storage
        upload_to_supabase(file_name, file_data)

        # Also provide the download button
        st.download_button(
            label="Download Predictions as Excel",
            data=file_data,
            file_name=file_name,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

