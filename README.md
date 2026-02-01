# 💎 Visualytics - Interactive Data Visualization Dashboard

## **Live Demo**

View Here: https://visualytics-interactive-data-visualization-dashboard-d4eknp6.streamlit.app/

Visualytics is an **interactive, professional-grade data visualization dashboard** built with **Python, Streamlit, Plotly, and Pandas**. It allows users to **explore, analyze, and visualize CSV datasets** instantly, with built-in filtering, KPIs, and predictive analytics.

---

## **🚀 Features**

### **1. Home Tab**
- Welcome interface with instructions.
- Highlights key dashboard capabilities:
  - Smart charts
  - KPIs & metrics
  - Dataset exploration

### **2. Dataset Tab**
- Preview dataset (top rows).
- Column information (numeric, text, date/time).
- Missing values overview (interactive bar chart).
- Filtered dataset view reflecting active sidebar filters.

### **3. Dashboard Tab**
- **Dynamic KPIs & Metrics**:
  - Total, Max, Min, Average, Std Deviation
  - Reflects current filters automatically

- **Charts & Visualizations**:
  - **Bar Chart** → compare numeric values by categories
  - **Clustered-Bar Chart** → compare multiple categories together
  - **Line Chart** → trends over time with optional forecasting
  - **Histogram** → distribution analysis (continuous & discrete)
  - **Pie Chart** → proportion analysis with top-N categories

- **Forecasting Feature**:
  - Holt-Winters Exponential Smoothing for future predictions
  - Automatic trend and seasonality detection
  - Customizable forecast periods

- **Interactive & Downloadable**:
  - All charts are interactive with Plotly
  - Charts can be downloaded as PNG

- **Smart Filters**:
  - Date, numeric, and categorical filtering
  - Multi-column filters with dynamic options

---

## **📦 Technologies Used**
- **Python** (Data processing & logic)
- **Streamlit** (Web dashboard)
- **Pandas & Numpy** (Data manipulation)
- **Plotly Express** (Interactive charts)
- **Statsmodels** (Time series forecasting)

---

## **📂 Usage Instructions**
1. Clone the repo:
git clone <repo-url>
2. Install dependencies:
pip install -r requirements.txt
3. Run the app:
streamlit run app.py
4. Upload a CSV file from the sidebar.

Explore KPIs, filters, and charts dynamically.

## 📂 Sample Datasets
To explore the dashboard features, you can use the included sample datasets:

- `sample_sales.csv` – Sales data with regions, products, and dates  
- `sample_marketing.csv` – Marketing campaign metrics by channels and dates  
- `sample_iot.csv` – IoT sensor readings with timestamps

The sample datasets used in this project are inspired by real-world data scenarios and sourced from publicly available datasets on Kaggle. These datasets were used for demonstration, analysis, visualization, and forecasting purposes.
- Kaggle (Customer Demographics & Shopping Preferences Data)
- Kaggle (Student Academic Performance Dataset)

Upload any of these datasets from the sidebar to immediately generate KPIs, charts, and forecasts.

💡 Notes

- CSV should ideally contain:
- At least one numeric column
- At least one text/categorical column
- Optional date/time column for trends and forecasting
- Cleaner datasets produce better insights.

🌟 Author

Unzila Sheikh – Data Enthusiast | Aspiring Data Analyst | Python & Visualization
