{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "04a1207e-d13a-4d05-9695-808eaf83cdfa",
   "metadata": {},
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# ==================================================\n",
    "# 1️⃣ LOAD MODEL & TRAINING COLUMNS\n",
    "# ==================================================\n",
    "model = joblib.load('ames_model.pkl')\n",
    "train_columns = pd.read_csv('train_columns.csv')['columns'].tolist()\n",
    "\n",
    "# ==================================================\n",
    "# 2️⃣ APP HEADER\n",
    "# ==================================================\n",
    "st.title(\"🏠 Ames Housing Price Predictor\")\n",
    "st.markdown(\"\"\"\n",
    "Enter the details of the house, and we'll predict its sale price.\n",
    "\"\"\")\n",
    "\n",
    "# ==================================================\n",
    "# 3️⃣ USER INPUTS\n",
    "# ==================================================\n",
    "user_inputs = {}\n",
    "\n",
    "user_inputs[\"LotArea\"] = st.number_input(\"Lot Area (sq ft):\", value=8000, step=100)\n",
    "user_inputs[\"OverallQual\"] = st.slider(\"Overall Quality (1–10):\", 1, 10, 5)\n",
    "user_inputs[\"YearBuilt\"] = st.number_input(\"Year Built:\", value=2000, step=1)\n",
    "\n",
    "# Optional Feature\n",
    "user_inputs[\"HouseAge\"] = 2024 - user_inputs[\"YearBuilt\"]\n",
    "\n",
    "# ==================================================\n",
    "# 4️⃣ BUILD USER DATAFRAME\n",
    "# ==================================================\n",
    "user_data = pd.DataFrame([user_inputs])\n",
    "\n",
    "# Ensure columns match training columns\n",
    "user_data = user_data.reindex(columns=train_columns, fill_value=0)\n",
    "\n",
    "# ==================================================\n",
    "# 5️⃣ PREDICTION\n",
    "# ==================================================\n",
    "if st.button(\"Predict Sale Price 💵\"):\n",
    "    prediction = model.predict(user_data)\n",
    "    st.markdown(f\"### 🏁 Predicted Sale Price: ${prediction[0]:,.2f}\")\n",
    "\n",
    "# ==================================================\n",
    "# 6️⃣ FEATURE IMPORTANCES\n",
    "# ==================================================\n",
    "st.markdown(\"---\")\n",
    "st.markdown(\"### 📊 Top 10 Feature Importances\")\n",
    "importances = pd.DataFrame({'Feature': train_columns, 'Importance': model.feature_importances_})\n",
    "importances = importances.sort_values(by='Importance', ascending=False)\n",
    "\n",
    "st.bar_chart(importances.head(10).set_index('Feature'))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
