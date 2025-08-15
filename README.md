# Apparel Dimension & Weight Predictor

This is a Streamlit-based web app for predicting the dimensions (Panjang, Lebar, Tinggi) and weight (Berat) of apparel based on product data.

## How to use the app

1. Clone or download this repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the app with `streamlit run app.py`.
4. Upload your Excel file to predict the dimensions and weight of products.

## Dependencies

- Streamlit
- pandas
- lightgbm
- numpy
- openpyxl
- scikit-learn

## Files

- **app.py**: Main code for the app.
- **model_folder**: Contains trained LightGBM models and the preprocessor.