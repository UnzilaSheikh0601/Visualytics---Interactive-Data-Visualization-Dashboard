import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dateutil import parser
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

#  -------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Visualytics - Charts Dashboard", 
    layout="wide"
)

st.title("💎 Visualytics - Interactive Data Visualization Dashboard")

#  -------------------------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Dashboard", "📂 Dataset"])

#  -------------------------------------------------------------------------------------------

def reset_filters():
    st.session_state.uploader_key += 1

    for key in list(st.session_state.keys()):
        if key.startswith(("bar_", "line_", "pie_", "clus_", "hist_", "met_", "filter_", "forecast_")):
            del st.session_state[key]

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if st.sidebar.button("🔄 Reset Dashboard"):
    reset_filters()
    st.rerun()

#  -------------------------------------------------------------------------------------------

st.sidebar.divider()

st.sidebar.header("📁 Upload & Controls")
st.sidebar.divider()

file = st.sidebar.file_uploader("Upload CSV File", type="csv", key=f"file_uploader_{st.session_state.uploader_key}")

#  -------------------------------------------------------------------------------------------

with tab1:
    st.title("👋 Welcome to the Analytics Dashboard")
    st.divider()

    if file is None:
        st.info("⬅️ Upload a CSV file from the sidebar to unlock the dashboard features.")
    else:
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.success("🎉 File Loaded Successfully!")
            with col2:
                st.info("➡️ Go to **Dashboard** or **Dataset** tabs for *Analysis*")

    st.divider()
    st.markdown("""
    ### 🚀 What you can do here:
    - Upload any CSV dataset  
    - Generate instant insights  
    - Create professional visualizations  
    - Download charts as images  
    """)
    st.divider()
    card_cont = st.container()
    with card_cont:
        card1, card2, card3 = st.columns(3)
        with card1:
            st.subheader("📊 Smart Dashboard")
            caption_cont = st.container(height=150)
            with caption_cont:
                st.caption("*Create powerful visuals in just a few clicks.*")
                st.write("Turn your raw data into clear, interactive charts that help you spot trends and patterns instantly.")
            st.markdown("""##### *Includes:*
- Bar charts for comparisons
- Line charts for trends over time
- Pie charts for proportions
- Histograms for distributions""")

        with card2:
            st.subheader("📈 KPIs & Metrics")
            caption_cont = st.container(height=150)
            with caption_cont:
                st.caption("*Get instant numerical insights without writing a single formula.*")
                st.write("Your important numbers are calculated automatically.")
            st.markdown("""##### *Track things like:*
- Total values
- Average performance
- Maximum & Minimum
- Standard Deviation (spread of data)""")
                
        with card3:
            st.subheader("📂 Dataset Explorer")
            caption_cont = st.container(height=150)
            with caption_cont:
                st.caption("*Understand your dataset before you analyze it.*")
                st.write("Explore its structure, quality, and column details in one place.")
            st.markdown("""##### *You can:*
- Preview rows of your dataset
- Check column data types
- Detect missing values
- Inspect filtered data live""")
    
    st.divider()
    csv_req = st.container()
    with csv_req:
        st.subheader("CSV Requirements")
        st.caption("*To unlock all features of the dashboard, your CSV file should ideally contain:*")
        st.markdown("""- At least one numeric column - used for charts, KPIs, and calculations
- At least one text/category column - used for grouping and comparisons
- A date or time column (optional) - enables trend analysis and forecasting""")
        st.info("⚠️ The cleaner your data (fewer missing values), the better the insights!")

#  --------------------------- IF EMPTY FILE -----------------------------------------------

if file is None:

    # -------------------------- DASHBOARD TAB -------------------------- 

    with tab2:

        st.title("📊 Interactive Dashboard")
        st.divider()

        st.info("⬅️ Upload a CSV file from the sidebar to unlock the dashboard features.")
        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            st.success("📌 KPIs & Metrics")
            st.write("View totals, averages, max & min values instantly.")

        with c2:
            st.success("📈 Dynamic Charts")
            st.write("Bar charts, line charts, histograms, and more.")

        st.divider()
        st.warning("Dashboard visuals will activate after dataset upload.")

        st.divider()
        corr_cont = st.container()
        st.subheader("🔗 Correlation Heatmap")
        st.info("💡 Upload dataset and explore relationship between variables")

    # -------------------------- DATASET TAB --------------------------

    with tab3:
        st.title("📂 Dataset Preview and Details")
        st.divider()

        st.info("⬅️ Upload a CSV file from the sidebar for dataset preview and details.")
        st.divider()

        with st.expander("🔍 What you'll see here"):
            st.write("""
            - Dataset preview  
            - Column data types  
            - Full data table view  
            - Easy inspection of structure  
            """)

        st.divider()

        st.caption("Upload a file to start exploring your data.")

#  --------------------------- IF NON-EMPTY FILE -----------------------------------------------

else:
    @st.cache_data(show_spinner=False)
    def load_data(file_bytes):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
        
    #  -------------------------------------------------------------------------------------------
        
    try:
        file_bytes = file.getvalue()
        data = load_data(file_bytes)
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    #  --------------------------- DATE COLUMNS FUNCTION -------------------------------------

    def robust_detect_date_columns(df, threshold=0.5):
        date_cols = []

        for col in df.columns:
            # Skip numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Already datetime?
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
                continue

            # Only object/string
            if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
                sample = df[col].dropna().astype(str).head(50)
                if sample.empty:
                    continue

                success_count = 0
                parsed_sample = []

                for val in sample:
                    try:
                        parsed_val = parser.parse(val, dayfirst=False)
                        parsed_sample.append(parsed_val)
                        success_count += 1
                    except:
                        parsed_sample.append(pd.NaT)

                success_rate = success_count / len(sample)

                if success_rate > threshold:
                    # Convert entire column
                    def safe_parse(x):
                        if pd.isnull(x):
                            return pd.NaT
                        try:
                            x_clean = str(x).replace(",", "")
                            return parser.parse(x_clean, dayfirst=False).replace(tzinfo=None)
                        except:
                            return pd.NaT  

                    df[col] = df[col].apply(safe_parse)
                    date_cols.append(col)

        return date_cols
    
    org_date_cols = robust_detect_date_columns(data)
    org_numeric_cols = data.select_dtypes(include=np.number).columns
    org_text_cols = data.select_dtypes(include=["object", "string"]).columns

    #  --------------------------- SELECTION FUNCTION -------------------------------------

    def selectbox_with_placeholder(label, options, key):
        options = list(options)

        if not options:
            st.sidebar.info(f"No valid options for {label}")
            return None

        options = ["--- Select ---"] + options
        selected = st.sidebar.selectbox(label, options, key=key)

        if selected == "--- Select ---":
            return None
        return selected

    #  --------------------------- DATA FILTERING -------------------------------------

    base_data = data.copy()      
    filtered_data = base_data.copy()

    st.sidebar.divider()
    st.sidebar.subheader("🔍 Filters")

    filter_cols = st.sidebar.multiselect(
    "Select Columns to Filter",
    base_data.columns,
    key="filter_col"
    )

    for filter_col in filter_cols:

        st.sidebar.markdown(f"#### 🔹 Filter: {filter_col.replace('_',' ').title()}")
    
        col_series = base_data[filter_col]   
    
        # DATE FILTER

        if filter_col in org_date_cols:
        
            col_series = pd.to_datetime(col_series, errors="coerce").dropna()
    
            if col_series.empty:
                st.sidebar.warning(f"No valid dates in {filter_col.replace('_',' ').title()}")
                continue
            
            min_date, max_date = col_series.min(), col_series.max()
    
            if pd.isna(min_date) or pd.isna(max_date):
                st.sidebar.warning(f"Invalid date values in {filter_col.replace('_',' ').title()}")
                continue
            
            date_range = st.sidebar.date_input(
                f"Date range for {filter_col.replace('_',' ').title()}",
                (min_date, max_date),
                key=f"filter_date_{filter_col}"
            )
    
            if len(date_range) == 2:
                filtered_data = filtered_data[
                    (pd.to_datetime(filtered_data[filter_col], errors="coerce") >= pd.to_datetime(date_range[0])) &
                    (pd.to_datetime(filtered_data[filter_col], errors="coerce") <= pd.to_datetime(date_range[1]))
                ]
    
        # NUMERIC FILTER 

        elif np.issubdtype(col_series.dtype, np.number):
        
            col_series = col_series.dropna()
    
            if col_series.empty:
                st.sidebar.warning(f"No numeric data in {filter_col.replace('_',' ').title()}")
                continue
            
            min_val, max_val = float(col_series.min()), float(col_series.max())
    
            if np.isnan(min_val) or np.isnan(max_val):
                st.sidebar.warning(f"Invalid numeric values in {filter_col.replace('_',' ').title()}")
                continue
            
            if min_val == max_val:
                st.sidebar.info(f"{filter_col.replace('_',' ').title()} has constant value {min_val}")
                continue
            
            selected_range = st.sidebar.slider(
                f"Range for {filter_col.replace('_',' ').title()}",
                min_val, max_val,
                (min_val, max_val),
                key=f"filter_range_{filter_col}"
            )
    
            filtered_data = filtered_data[
                (filtered_data[filter_col] >= selected_range[0]) &
                (filtered_data[filter_col] <= selected_range[1])
            ]
    
        # TEXT FILTER 

        else:
        
            unique_vals = col_series.dropna().unique()
    
            if len(unique_vals) == 0:
                st.sidebar.warning(f"No values in {filter_col.replace('_',' ').title()}")
                continue
            
            if len(unique_vals) > 200:
                unique_vals = unique_vals[:200]
                st.sidebar.info("Showing first 200 values")
    
            values = st.sidebar.multiselect(
                f"Values for {filter_col.replace('_',' ').title()}",
                unique_vals,
                default=unique_vals,
                key=f"filter_values_{filter_col}"
            )
    
            filtered_data = filtered_data[filtered_data[filter_col].isin(values)]

    numeric_cols = filtered_data.select_dtypes(include=np.number).columns
    text_cols = filtered_data.select_dtypes(include=["object", "string"]).columns
    date_cols = filtered_data.select_dtypes(include=["datetime64[ns]"]).columns

    #  --------------------------- EMPTY FILTERS FUNCTION -------------------------------------

    def emp_fil(fil_cols):
        if fil_cols.empty:
            st.info("No data available after filtering. Try valid filtering!")
            return
        return True

    #  -------------------------------------------------------------------------------------------

    # -------------------------- DATASET TAB --------------------------

    with tab3:
        st.title("📂 Dataset Preview and Details")
        st.divider()
        
        data_info = st.container()
        with data_info:
            st.subheader("📊 Dataset Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", data.shape[0])
            c2.metric("Columns", data.shape[1])
            c3.metric("Missing Values", int(data.isnull().sum().sum()))
            st.divider()

            col1, col2= st.columns(2)
            with col1:
                st.subheader("Dataset Columns")
                with st.expander("View"):
                    for cols in data.columns:
                        st.markdown(f"- {cols}")            

            with col2:
                st.subheader("Column Info")
                with st.expander("view"):
                    if len(org_text_cols) == 0:
                        st.info("No text columns found in dataset")
                    else:
                        st.info(f"{len(org_text_cols)} text columns found in dataset")
                    if len(org_numeric_cols) == 0:
                        st.info("No numeric columns found in dataset")
                    else:
                        st.info(f"{len(org_numeric_cols)} numeric columns found in dataset")
                    if len(org_date_cols) == 0:
                        st.info("No date/time columns found in dataset")
                    else:
                        st.info(f"{len(org_date_cols)} date/time columns found in dataset")
        st.divider()

        preview_container = st.container()
        filtered_container = st.container()

        with preview_container:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🔍 Dataset Preview")
                st.dataframe(data.head(10))

            with c2:
                st.subheader("🧩 Missing Values Overview")

                missing = data.isnull().sum()
                missing = missing[missing > 0]

                if len(missing) > 0:
                    fig = px.bar(
                        missing,
                        title="Missing Values Per Column",
                        labels={"value": "Missing Count", "index": "Columns"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Overview of original dataset ")
                else:
                    st.success("No missing values in dataset 🎉")

            st.divider()

        with filtered_container:
            st.subheader("📂 View Filtered Dataset")
            st.caption("Preview reflects active sidebar filters.")
            with st.expander("View Filtered Dataset"):
                if emp_fil(filtered_data) == True:
                    st.dataframe(filtered_data)

    # -------------------------- DASHBOARD TAB --------------------------

    with tab2:
        st.title("📊 Interactive Charts Dashboard")
        st.divider()

        col1, col2= st.columns([1, 2.5])

        metrics_container = col1.container()
        chart_container = col2.container()

    #  --------------------------- METRICS SECTION -------------------------------------

        st.sidebar.divider()
        st.sidebar.subheader("📌 Key Metrics")
        met_col = selectbox_with_placeholder("Select Key Metrics", numeric_cols, key="met_col")

        with metrics_container:
            st.subheader("📌 Key Metrics")
            st.success("📌 KPIs & Metrics")
            st.divider()

            if met_col is not None and met_col in filtered_data.columns:
                met_fil = filtered_data[met_col]
                if not met_fil.empty:
                    st.caption("Metrics reflect current filters applied.")
                    st.metric(f"🔢 Total of {met_col.replace('_',' ').title()}", f"{filtered_data[met_col].sum():,.0f}")
                    st.metric(f"📈 Max of {met_col.replace('_',' ').title()}", f"{filtered_data[met_col].max():,.0f}")
                    st.metric(f"📉 Min of {met_col.replace('_',' ').title()}", f"{filtered_data[met_col].min():,.0f}")
                    st.metric(f"📊 Average of {met_col.replace('_',' ').title()}", f"{filtered_data[met_col].mean():,.2f}")
                    st.metric(f"🧮 Std Deviation of {met_col.replace('_',' ').title()}", f"{filtered_data[met_col].std():,.2f}")
                else:
                    st.info("No data available after filtering. Try valid filtering!")
            else:
                st.info("Please select a valid metric column")

        st.divider()

    #  --------------------------- CORRELATION HEATMAP -------------------------------------

        st.subheader("🔗 Correlation Heatmap")
        st.divider()

        constant_cols = [c for c in numeric_cols if filtered_data[c].nunique() <= 1]
        if constant_cols:
            st.info(f"Skipped constant columns: {', '.join(constant_cols.replace('_',' ').title())}")

        if len(numeric_cols) > 1 and len(filtered_data) > 1:
        
            corr = filtered_data[numeric_cols].corr()
            corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")

            if corr.shape[0] < 2:
                st.info("Not enough valid numeric relationships to compute correlation.")
            else:
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Between Numeric Columns",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Need at least 2 numeric columns and 2 data rows for correlation.")

    #  --------------------------- CHART INPUTS -------------------------------------

        st.sidebar.divider()
        st.sidebar.subheader("📊 Charts")
        chart_type = selectbox_with_placeholder("Select the type of visual",
                        ["Bar Chart", "Clustered-Bar Chart", "Line Chart", "Histogram", "Pie Chart"],
                        key="chart_type")    

        with chart_container:

            c1, c2, c3 = st.columns(3)
            visual_head = c1.container()
            save_head = c2.container()
            chart_head = c3.container()

            with visual_head:
                st.subheader("📊 Visual")
            st.success("📈 Dynamic Charts")
            st.divider()

    #  --------------------------- CHART WARNINGS -------------------------------------

            data_warn = st.container()
            
            with data_warn:
                if chart_type in ["Bar Chart", "Pie Chart", "Clustered-Bar Chart"] and (len(text_cols) == 0 or len(numeric_cols) == 0):
                    st.error("⚠️ This chart requires at least one text column and one numeric column")
                    st.stop()
                elif chart_type == "Line Chart" and (len(date_cols) == 0 or len(numeric_cols) == 0):
                    st.error("⚠️ No date/time or numeric columns found in dataset")  
                    st.stop()
                elif chart_type == "Histogram" and len(numeric_cols) == 0:
                    st.error("⚠️ No numeric columns found in dataset")  
                    st.stop()

    #  --------------------------- AGGREGATION & CHART DOWNLOAD FUNCTION -------------------------------------

            def aggregate_data(df, group_cols, value_col, agg_func):
                agg_map = {
                    "Sum": "sum",
                    "Average": "mean",
                    "Count": "count"
                }
                return df.groupby(group_cols)[value_col].agg(agg_map[agg_func])
            
            agg_title_map = {
                "Sum": "Total",
                "Average": "Average",
                "Count": "Count of"
            }

            
            def download_chart(fig, name):
                buf = io.BytesIO()
                try:
                    fig.write_image(buf, format="png")
                    buf.seek(0)
                    with save_head:
                        st.download_button(
                            "📥 Download Chart (PNG)",
                            buf,
                            file_name=f"{name}.png",
                            mime="image/png"
                        )
                except:
                    st.warning("Image download requires kaleido package.")

    #  --------------------------- BAR CHART -------------------------------------

            def plot_bar_chart(df):
                x_axis = selectbox_with_placeholder("X-axis", text_cols, key="bar_x")
                y_axis = selectbox_with_placeholder("Y-axis", numeric_cols, key="bar_y")
                agg_func = selectbox_with_placeholder("Aggregation", ["Sum","Average","Count"], key="bar_agg")

                with chart_head:
                    st.subheader("📊 Bar Chart")

                if not x_axis or not y_axis or not agg_func:
                    st.info("Select all fields")
                    return

                grouped = aggregate_data(df, x_axis, y_axis, agg_func).sort_values(ascending=False)
                grouped = grouped.reset_index()

                friendly_agg = agg_title_map.get(agg_func, agg_func)
                title_text = f"{friendly_agg} {y_axis.replace('_',' ').title()} by {x_axis.replace('_',' ').title()}"

                if emp_fil(grouped) == True:

                    fig = px.bar(grouped, x=x_axis, y=y_axis, text_auto=True,
                                 title=title_text)
                    
                    fig.update_layout(
                        xaxis_title=x_axis.replace("_", " ").title(),
                        yaxis_title=y_axis.replace("_", " ").title()
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    download_chart(fig, "bar_chart")

    #  --------------------------- CLUSTERED-BAR CHART -------------------------------------

            def plot_clustered_bar(df):
                x_axis = selectbox_with_placeholder("Main Category", text_cols, key="clus_x")
                cluster_col = selectbox_with_placeholder("Cluster Category", text_cols, key="clus_cluster")
                y_axis = selectbox_with_placeholder("Value", numeric_cols, key="clus_y")
                agg_func = selectbox_with_placeholder("Aggregation", ["Sum","Average","Count"], key="clus_agg")

                with chart_head:
                    st.subheader("📊 Clustered-Bar Chart")

                if not x_axis or not cluster_col or not y_axis or not agg_func:
                    st.info("Select all fields")
                    return
                
                if df[x_axis].nunique() > 100 or df[cluster_col].nunique() > 50:
                    st.warning("Too many categories for clustered bar chart.")
                    return

                grouped = aggregate_data(df, [x_axis, cluster_col], y_axis, agg_func).unstack().fillna(0)
                grouped = grouped.reset_index()

                friendly_agg = agg_title_map.get(agg_func, agg_func)
                title_text = f"{friendly_agg} {y_axis.replace('_',' ').title()} by {x_axis.replace('_',' ').title()} and {cluster_col.replace('_',' ').title()}"

                if emp_fil(grouped) == True:

                    y_cols = grouped.columns.difference([x_axis]).tolist()

                    fig = px.bar(grouped, x=x_axis, y=y_cols, barmode="group",
                                 title=title_text)
                    
                    fig.update_layout(
                        xaxis_title=f"{x_axis.replace('_', ' ').title()} and {cluster_col.replace('_',' ').title()}",
                        yaxis_title=y_axis.replace("_", " ").title()
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    download_chart(fig, "clustered_bar")

    #  --------------------------- LINE CHART -------------------------------------

            def plot_line_chart(df):
                x_axis = selectbox_with_placeholder("Date Column", date_cols, key="line_x")
                y_axis = selectbox_with_placeholder("Value Column", numeric_cols, key="line_y")
                agg_func = selectbox_with_placeholder("Aggregation", ["Sum","Average","Count"], key="line_agg")
                freq = selectbox_with_placeholder("Time Interval", ["Hour", "Day","Week","Month","Year"], key="line_freq")

                with chart_head:
                    st.subheader("📈 Line Chart")

                if not x_axis or not y_axis or not agg_func or not freq:
                    st.info("Select all fields")
                    return

                freq_map = {"Hour":"H", "Day":"D","Week":"W","Month":"M","Year":"Y"}
                df = df.sort_values(x_axis)

                df = df.dropna(subset=[x_axis])
                if not pd.api.types.is_datetime64_any_dtype(df[x_axis]):
                    st.error("Selected date column is not valid datetime.")
                    return

                grouped = df.set_index(x_axis).resample(freq_map[freq])[y_axis] \
                            .agg({"Sum":"sum","Average":"mean","Count":"count"}[agg_func]) \
                            .reset_index()
                friendly_agg = agg_title_map.get(agg_func, agg_func)
                title_text = f"{friendly_agg} {y_axis.replace('_',' ').title()} by {x_axis.replace('_',' ').title()}"
                
                if emp_fil(grouped) == True:
                
                    fig = px.line(grouped, x=x_axis, y=y_axis, markers=True,
                                  title=title_text)
                    fig.update_layout(
                        xaxis_title=x_axis.replace('_', ' ').title(),
                        yaxis_title=y_axis.replace('_', ' ').title()
                    )

                    c1, c2 = st.columns([1, 2])

                    with c1:
                        forecast_toggle = st.checkbox("🔮 Enable Forecasting")

                    if forecast_toggle:
                    
                        # Use user-selected frequency directly
                        freq_value = freq_map[freq]
                        freq_labels = {"H": "Hour", "D": "Days", "W": "Weeks", "M": "Months", "Y": "Years"}

                        data_points = len(grouped)
                        max_periods = data_points // 2

                        if max_periods < 5:
                            with c2:
                                st.error("Not enough data to generate forecast.")
                        else:
                            periods = st.slider(
                                f"Forecast periods ({freq_labels.get(freq_value, 'Steps')})",
                                min_value=5,
                                max_value=max_periods,
                                value=5,
                                key="forecast_periods"
                            )

                            # ---- Seasonality Rules ----
                            seasonal_type, seasonal_periods = None, None

                            if freq_value.startswith("M"):      # Monthly
                                seasonal_type, seasonal_periods = "add", 12
                            elif freq_value == "W":             # Weekly
                                seasonal_type, seasonal_periods = "add", 52
                            elif freq_value == "D":             # Daily
                                seasonal_type, seasonal_periods = "add", 7

                            # Disable seasonality if not enough history
                            if seasonal_periods and data_points < seasonal_periods * 2:
                                seasonal_type, seasonal_periods = None, None
                                st.info("Seasonality disabled (not enough history). Using trend-only forecast.")

                            # ---- Build Model ----
                            try:
                                if seasonal_type:
                                    model = ExponentialSmoothing(
                                        grouped[y_axis],
                                        trend="add",
                                        seasonal=seasonal_type,
                                        seasonal_periods=seasonal_periods
                                    ).fit()
                                else:
                                    model = ExponentialSmoothing(grouped[y_axis], trend="add").fit()

                            except Exception as e:
                                st.error(f"Forecast failed: {e}")
                                return

                            # ---- Forecast ----
                            forecast = model.forecast(periods)

                            last_date = grouped[x_axis].max()
                            future_dates = pd.date_range(
                                start=last_date + pd.tseries.frequencies.to_offset(freq_value),
                                periods=periods,
                                freq=freq_value
                            )

                            fig.add_scatter(
                                x=future_dates,
                                y=forecast,
                                mode="lines+markers",
                                name="Forecast",
                                line=dict(dash="dash")
                            )

                    st.plotly_chart(fig, use_container_width=True)
                    download_chart(fig, "line_chart")

    #  --------------------------- HISTOGRAM -------------------------------------

            def plot_histogram(df):
                col = selectbox_with_placeholder("Numeric Column", numeric_cols, key="hist_col")

                with chart_head:
                    st.subheader("📊 Histogram")

                if not col:
                    st.info("Select column")
                    return

                series = df[col].dropna()
                
                if emp_fil(series) == True:

                    bins = st.slider("Bins", 5, 20, 10, key="hist_bins")
                    is_discrete = df[col].nunique() < 25 and pd.api.types.is_integer_dtype(df[col])

                    if is_discrete:
                        counts = df[col].value_counts().sort_index()

                        fig = px.bar(
                            x=counts.index,
                            y=counts.values,
                            labels={"x": col, "y": "Count"},
                            title=f"Discrete Distribution of {col.replace('_', ' ').title()}"
                        )
                    else:
                        fig = px.histogram(df, x=col, nbins=bins, title=f"Continuous Distribution of {col.replace('_', ' ').title()}")

                    fig.update_traces(
                        marker=dict(line=dict(color="black", width=0.5)),
                        opacity=0.85
                    )
                    fig.update_layout(
                        xaxis_title=col.replace('_', ' ').title(),
                        yaxis_title="Count"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    download_chart(fig, "histogram")

    #  --------------------------- PIE CHART -------------------------------------

            def plot_pie_chart(df):
                labels = selectbox_with_placeholder("Category", text_cols, key="pie_label")
                values = selectbox_with_placeholder("Value", numeric_cols, key="pie_val")
                agg_func = selectbox_with_placeholder("Aggregation", ["Sum","Average","Count"], key="pie_agg")

                with chart_head:
                    st.subheader("◔ Pie Chart")

                if not labels or not values or not agg_func:
                    st.info("Select all fields")
                    return

                grouped = aggregate_data(df, labels, values, agg_func).sort_values(ascending=False).head(10).reset_index()
                friendly_agg = agg_title_map.get(agg_func, agg_func)
                title_text = f"{friendly_agg} {values.replace('_',' ').title()} by {labels.replace('_',' ').title()}"
                if emp_fil(grouped) == True:

                    fig = px.pie(grouped, names=labels, values=values,
                                 title=title_text)
                    st.plotly_chart(fig, use_container_width=True)
                    download_chart(fig, "pie_chart")

    #  -------------------------------------------------------------------------------------------

            chart_map = {
                "Bar Chart": plot_bar_chart,
                "Clustered-Bar Chart": plot_clustered_bar,
                "Line Chart": plot_line_chart,
                "Histogram": plot_histogram,
                "Pie Chart": plot_pie_chart
            }

            if chart_type in chart_map:
                chart_map[chart_type](filtered_data)
            else:
                st.info("Select a chart type")
            
#  -------------------------------------------------------------------------------------------