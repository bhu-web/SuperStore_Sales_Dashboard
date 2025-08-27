# Importing necessary libraries for data manipulation, visualization, and forecasting
import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
from plotly import graph_objects as go
from st_aggrid import AgGrid
import plotly.figure_factory as ff
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from pmdarima import auto_arima

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore') 
# Set the Streamlit app configuration
st.set_page_config(page_title="Superstore EDA", page_icon=":bar_chart:", layout="wide"
)
# Set the app's main title and subtitle
st.title(":bar_chart: Superstore EDA")
st.write("*An interactive app for exploratory data analysis and forecasting.*")

# Function to display various insights about the dataset
def display_data_insights(df):
    # Define available insight options
    insights_options = ["Basic Info",
                        "Preview of the Data",
                        "Missing Values", 
                        "Duplicate Rows", 
                        "Numeric Column Statistics", 
                        "Categorical Column Value Counts", 
                        "Correlation Heatmap"]
    st.markdown("<h2>Select Insight to Display</h2>", unsafe_allow_html=True)
    selected_insight = st.selectbox(
        "Select insight:", insights_options, label_visibility="collapsed"
    )

    if selected_insight == "Basic Info":
        st.write("## Basis Info")
        st.write("### Dataset Insights")
        st.write(f"**Number of Rows:** {df.shape[0]}")
        st.write(f"**Number of Columns:** {df.shape[1]}")
        st.write("### Column Details")
        st.dataframe(df.dtypes.astype(str).rename("Data Type"))
    elif selected_insight == "Preview of the Data":
        st.write("### Preview of the Data")
        st.dataframe(df.head())
    elif selected_insight == "Missing Values":
        missing_values = df.isnull().sum()
        st.write("### Missing Values")
        st.write(f"**Total Missing Values by Column:**")
        st.dataframe(missing_values[missing_values > 0].rename("Missing Count"))
        rows_with_missing = df[df.isnull().any(axis=1)]
        st.write("### Rows with Missing Values")
        st.dataframe(rows_with_missing)
    elif selected_insight == "Duplicate Rows":
        duplicates = df.duplicated().sum()
        st.write(f"**Number of Duplicate Rows:** {duplicates}")
        duplicate_rows = df[df.duplicated()]
        st.write("### Duplicate Rows")
        st.dataframe(duplicate_rows)
    elif selected_insight == "Numeric Column Statistics":
        st.write("### Numeric Column Statistics")
        st.dataframe(df.describe().transpose())
    elif selected_insight == "Categorical Column Value Counts":
        st.write("### Categorical Column Value Counts")
        categorical_columns = df.select_dtypes(include="object").columns
        for col in categorical_columns:
            with st.expander(f"**{col}** - Value Counts"):
                st.dataframe(df[col].value_counts())
    elif selected_insight == "Correlation Heatmap":
        st.write("### Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        st.write(corr.style.background_gradient(cmap="coolwarm"))

# Function to handle missing values in the dataset
def handle_missing_values(df, missing_action):
    if missing_action == "Drop Rows with Missing Values":
        df = df.dropna()
    elif missing_action == "Fill Missing Values with Mean (Numeric Columns)":
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_columns:
            df[col].fillna(df[col].mean(), inplace=True)
    elif missing_action == "Fill Missing Values with Median (Numeric Columns)":
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)
    elif missing_action == "Fill Missing Values with Mode (Categorical Columns)":
        categorical_columns = df.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    elif missing_action == "Fill Missing Values with Custom Value":
        custom_value = st.text_input("Enter a custom value to fill missing values:", "Unknown")
        df.fillna(custom_value, inplace=True)
    return df

# Function to handle duplicate values in the dataset
def handle_duplicate_values(df, duplicate_action):
    if duplicate_action == "Keep All Rows":
        pass  
    elif duplicate_action == "Drop Duplicate Rows (Keep First Instance)":
        df = df.drop_duplicates(keep="first")  
    elif duplicate_action == "Drop Duplicate Rows (Keep Last Instance)":
        df = df.drop_duplicates(keep="last")  
    elif duplicate_action == "Mark Duplicates":
        df["Is Duplicate"] = df.duplicated(keep=False)  
    return df

# Function to filter data based on user-selected location (Region, State, City)
def filter_by_location(df, region, state, city):
    if not region and not state and not city:
        return df  
    elif not state and not city:
        return df[df["Region"].isin(region)]  
    elif not region and not city:
        return df[df["State"].isin(state)]  
    elif state and city:
        return df[df["State"].isin(state) & df["City"].isin(city)]  
    elif region and city:
        return df[df["Region"].isin(region) & df["City"].isin(city)]  
    elif region and state:
        return df[df["Region"].isin(region) & df["State"].isin(state)]  
    elif city:
        return df[df["City"].isin(city)]  
    else:
        return df[df["Region"].isin(region) & df["State"].isin(state) & df["City"].isin(city)]  

# Function to generate various types of charts based on input parameters
def create_chart(df, chart_type, x_column, y_column, title, color_column=None):
    fig = None
    if chart_type == "Bar Chart":
        fig = px.bar(df, x=x_column, y=y_column, title=title, color=color_column)
        # Calculate insights for Bar Chart
        total_sales = df[y_column].sum()
        average_sales = df[y_column].mean()
        st.write(f"**Total {y_column}:** {total_sales}")
        st.write(f"**Average {y_column}:** {average_sales}")
    elif chart_type == "Pie Chart":
        fig = px.pie(df, values=y_column, names=x_column, title=title)
        # Calculate insights for Pie Chart
        pie_data = df.groupby(x_column)[y_column].sum()
        largest_segment = pie_data.idxmax()
        largest_value = pie_data.max()
        st.write(f"**Largest Segment:** {largest_segment} with {largest_value} {y_column}")
    elif chart_type == "Line Chart":
        # Group by x_column and sum the y_column (sales)
        line_df = df.groupby(x_column, as_index=False)[y_column].sum()
        # Sort the DataFrame by the x_column (for better line chart plotting)
        line_df = line_df.sort_values(by=x_column)
        fig = px.line(line_df, x=x_column, y=y_column, title=title)
        # Calculate insights for Line Chart
        total_sales = line_df[y_column].sum()
        average_sales = line_df[y_column].mean()
        growth_rate = (line_df[y_column].iloc[-1] - line_df[y_column].iloc[0]) / line_df[y_column].iloc[0] * 100
        st.write(f"**Total {y_column}:** {total_sales}")
        st.write(f"**Average {y_column}:** {average_sales}")
        st.write(f"**Growth Rate in {y_column}:** {growth_rate:.2f}%")
    return fig

# Function to plot time-series sales data
def plot_time_series_sales(df):
    df["month_year"] = df["Order Date"].dt.to_period("M")  
    linechart = df.groupby(df["month_year"].dt.strftime("%Y-%b"))["Sales"].sum().reset_index()  
    fig = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"})  
    # Add trendline to the plot    
    fig.update_traces(mode='lines+markers', line=dict(shape='linear'))
    return fig 
# Function to generate insights from time series sales data
def generate_sales_insights(df):
    # Group by month-year and calculate total sales per month
    df["month_year"] = df["Order Date"].dt.to_period("M")
    monthly_sales = df.groupby(df["month_year"].dt.strftime("%Y-%b"))["Sales"].sum().reset_index()
    # Total sales over the period
    total_sales = monthly_sales["Sales"].sum()
    # Best performing month
    best_month = monthly_sales.loc[monthly_sales["Sales"].idxmax()]
    best_month_value = best_month["Sales"]
    best_month_name = best_month["month_year"]
    # Worst performing month
    worst_month = monthly_sales.loc[monthly_sales["Sales"].idxmin()]
    worst_month_value = worst_month["Sales"]
    worst_month_name = worst_month["month_year"]
    # Month-over-Month growth calculation
    monthly_sales["prev_sales"] = monthly_sales["Sales"].shift(1)
    monthly_sales["growth"] = ((monthly_sales["Sales"] - monthly_sales["prev_sales"]) / monthly_sales["prev_sales"]) * 100
    latest_growth = monthly_sales["growth"].iloc[-1]
    # Insights as a dictionary
    insights = {
        "total_sales": total_sales,
        "best_month": best_month_name,
        "best_month_value": best_month_value,
        "worst_month": worst_month_name,
        "worst_month_value": worst_month_value,
        "latest_growth": latest_growth,
    }
    return insights

# Function to plot sales in a treemap
def plot_sales_treemap(df):
    fig = px.treemap(df, path=["Region", "Category", "Sub-Category"], values="Sales", color="Sub-Category")  # Create treemap
    return fig  

# Function to plot a scatter plot for sales vs. profit
def plot_scatter_sales_profit(df):
    fig = px.scatter(df, x="Sales", y="Profit", size="Quantity", color="Category",
                     hover_data=["Region", "State", "City", "Sub-Category", "Order ID", "Sales", "Profit", "Quantity"])  # Create scatter plot
    return fig  
# Function to generate insights from sales and profit data grouped by Category
def generate_sales_profit_insights_by_category(df):
    category_grouped = df.groupby("Category").agg(
        total_sales=("Sales", "sum"),
        total_profit=("Profit", "sum"),
        avg_sales_profit_ratio=("Sales", lambda x: (x.sum() / x.replace(0, 1).sum())),
        correlation=("Sales", lambda x: x.corr(df.loc[x.index, "Profit"])),
    ).reset_index()
    # Top 5 products by sales and profit per category
    top_sales_by_category = df.groupby(["Category", "Sub-Category"])["Sales"].sum().reset_index()
    top_profit_by_category = df.groupby(["Category", "Sub-Category"])["Profit"].sum().reset_index()
    # Get top 5 products (by sales) for each category
    top_sales_by_category = top_sales_by_category.sort_values(["Category", "Sales"], ascending=[True, False]).groupby("Category").head(5)
    # Get top 5 products (by profit) for each category
    top_profit_by_category = top_profit_by_category.sort_values(["Category", "Profit"], ascending=[True, False]).groupby("Category").head(5)
    # Identify outliers (high sales, low profit) by category
    outliers_by_category = df[(df["Sales"] > df["Sales"].quantile(0.95)) & (df["Profit"] < df["Profit"].quantile(0.05))]
    insights = {
        "category_grouped": category_grouped,
        "top_sales_by_category": top_sales_by_category,
        "top_profit_by_category": top_profit_by_category,
        "outliers_by_category": outliers_by_category
    }
    return insights

# --- Sales Forecasting using Auto ARIMA ---
def forecast_sales(data, forecast_period=12):
    try:
        data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
        sales_data = data.groupby('Order Date')['Sales'].sum()

        # Fit auto_arima model (non-seasonal for now)
        model = auto_arima(
            sales_data,
            seasonal=False,
            trace=True,  # show search process in logs
            error_action="ignore",  # ignore errors and keep searching
            suppress_warnings=True
        )
        
        # Forecast
        forecast = model.predict(n_periods=forecast_period)

        # Generate future dates for the forecast period
        last_date = sales_data.index[-1]
        future_dates = pd.date_range(last_date, periods=forecast_period + 1, freq='M')[1:]
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Sales': forecast
        })

        # Plot the forecast
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=sales_data.index, y=sales_data.values,
            mode='lines', name='Historical Sales'
        ))
        forecast_fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Forecasted Sales'],
            mode='lines', name='Forecasted Sales',
            line=dict(dash='dash')
        ))
        forecast_fig.update_layout(
            title="Sales Forecast (Auto ARIMA)",
            xaxis_title="Date",
            yaxis_title="Sales"
        )
        st.plotly_chart(forecast_fig)

    except Exception as e:
        st.error(f"An error occurred while forecasting: {e}")
        st.warning("Try adjusting the dataset or forecast period.")

# Function to generate visualizations with error handling
def create_charts(data):
    try:
        col1, col2, col3 = st.columns(3)
        # Sales by Category (Bar chart)
        with col1:
            if 'Category' in data.columns and 'Sales' in data.columns:
                fig = px.bar(data, x="Category", y="Sales", title="Sales by Category")  # Sales by Category chart
                st.plotly_chart(fig, use_container_width=True)  # Display the chart
            else:
                st.warning("Required columns for 'Sales by Category' chart are missing.")
        # Sales by Region (Bar chart)
        with col2:
            if 'Region' in data.columns and 'Sales' in data.columns:
                fig2 = px.bar(data, x="Region", y="Sales", title="Sales by Region")  # Sales by Region chart
                st.plotly_chart(fig2, use_container_width=True)  # Display the chart
            else:
                st.warning("Required columns for 'Sales by Region' chart are missing.")
        # Profit by Category (Bar chart)
        with col3:
            if 'Category' in data.columns and 'Profit' in data.columns:
                fig3 = px.bar(data, x="Category", y="Profit", title="Profit by Category")  # Profit by Category chart
                st.plotly_chart(fig3, use_container_width=True)  # Display the chart
            else:
                st.warning("Required columns for 'Profit by Category' chart are missing.")
    except Exception as e:
        st.error(f"An error occurred while generating charts: {e}")  # Error handling during chart creation

# Function to generate downloadable data as CSV
def create_download_button(df, columns, file_name):
    csv_data = df[columns].to_csv(index=False)
    st.download_button(
        label=f"Download {file_name}",
        data=csv_data,
        file_name=f"{file_name}.csv",
        mime="text/csv"
    )

# Tabs for organizing sections
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Data Cleaning", "Visualizations", "Forecasting"])

# Tab 1: Dataset
with tab1:
    st.header("Upload Your Dataset")
    # File uploader widget to allow user to upload a file
    fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "txt", "xlsx", "xls"])
    # Function to load the dataset from the uploaded file (with caching to avoid reloading every time)
    @st.cache_data
    def load_data(file):
        try:
            # Read the file into a pandas DataFrame
            df = pd.read_excel(file)  # You can switch this to pd.read_csv(file) if uploading a CSV
            return df
        except Exception as e:
            # If an error occurs, display an error message
            st.error(f"Error reading the file: {e}")
            st.stop()
    # Load the data if file is uploaded, otherwise load a default file from the local machine
    if fl is not None:
        df = load_data(fl)
        st.success("File successfully uploaded!")
        display_data_insights(df)  # Load the dataset from uploaded file
    else:
        st.error("Please upload a dataset to begin.")  
        st.stop()

# Tab 2: Data Cleaning
with tab2:
    st.header("Data Cleaning")
    # Summarize missing values
    st.write("### Missing Values")
    missing_summary = df.isnull().sum().reset_index()
    missing_summary.columns = ["Column", "Missing Values"]
    missing_summary["% Missing"] = (missing_summary["Missing Values"] / len(df)) * 100
    # Check if there are any missing values
    if missing_summary["Missing Values"].sum() > 0:
        st.write("### Missing Values Detected")
        st.write(missing_summary.style.background_gradient(cmap="Reds"))
        # Display rows with missing values
        st.write("### Rows with Missing Values")
        st.dataframe(df[df.isnull().any(axis=1)])
        # User options for handling missing values
        missing_action = st.radio(
            "How would you like to handle missing values?",
            [
                "Drop Rows with Missing Values",
                "Fill Missing Values with Mean (Numeric Columns)",
                "Fill Missing Values with Median (Numeric Columns)",
                "Fill Missing Values with Mode (Categorical Columns)",
                "Fill Missing Values with Custom Value",
            ],
        )
        # Handle missing values based on user input
        df = handle_missing_values(df, missing_action)
        st.write(f"#### Missing values handled: {missing_action}")
    else:
        st.write("#### No Missing Values Detected")
    # Handle 'NaN' values in 'Category' and 'Sub-Category' for visualization purposes (treemap)
    df['Category'] = df['Category'].fillna('Unknown')  # Fill missing 'Category' values with 'Unknown'
    df['Sub-Category'] = df['Sub-Category'].fillna('Unknown')  # Fill missing 'Sub-Category' values with 'Unknown'
    # Calculate duplicate count
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.write(f"### {duplicate_count} Duplicate Rows Detected")
        # Display rows that are duplicated
        st.write("### Duplicate Rows")
        st.dataframe(df[df.duplicated()])
        # User options for handling duplicate values
        duplicate_action = st.radio(
            "How would you like to handle duplicate values?",
            [
                "Keep All Rows",
                "Drop Duplicate Rows (Keep First Instance)",
                "Drop Duplicate Rows (Keep Last Instance)",
                "Mark Duplicates",
            ],
        )
        # Handle duplicate rows based on user input
        df = handle_duplicate_values(df, duplicate_action)
        st.write(f"#### Duplicate values handled: {duplicate_action}")
    else:
        st.write("#### No Duplicate Rows Detected")
        # Display the cleaned data to the user
    st.subheader("Raw Data")
    data_display_option = st.radio(
    "Choose how to view the raw data:",  # Allow the user to choose how to view the raw data
    options=["Display All Rows", "Display First 100 Rows", "Interactive Table"],  # Options to display the data
    horizontal=True  # Display options horizontally
    )
    # Display the raw data based on the user's selection
    if data_display_option == "Display All Rows":
        st.dataframe(df)  # Show all rows of the cleaned data
    elif data_display_option == "Display First 100 Rows":
        st.dataframe(df.head(100))  # Show the first 100 rows of the cleaned data
    else:
        AgGrid(df)  # Display an interactive table for the raw data (if AgGrid is installed)

# Sidebar for filtering options
st.sidebar.header("**Data Filtering**")
# Date Filtering Section in Sidebar
st.sidebar.subheader("*Filter by Date*")
# Convert 'Order Date' column to datetime format to enable time-based filtering
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
# Date range filtering options
startDate = df["Order Date"].min()  # Default start date as the minimum order date in the data
endDate = df["Order Date"].max()  # Default end date as the maximum order date in the data
# Date input widgets for filtering the data based on user input
date1 = st.sidebar.date_input("Start Date", startDate)  # Start date input
date2 = st.sidebar.date_input("End Date", endDate)  # End date input
# Filter the data based on the selected date range
df = df[(df["Order Date"] >= pd.to_datetime(date1)) & (df["Order Date"] <= pd.to_datetime(date2))]
# Location Filtering Section in Sidebar
st.sidebar.subheader("*Filter by Location*")
# Region filter
region = st.sidebar.multiselect(
    "Pick your Region", 
    options=df["Region"].unique(), 
    help="Filter data by region."
)
# State filter (dependent on Region selection)
state = st.sidebar.multiselect(
    "Pick the State", 
    options=df[df["Region"].isin(region)]["State"].unique() if region else df["State"].unique(), 
    help="Filter data by state."
)
# City filter (dependent on State selection)
city = st.sidebar.multiselect(
    "Pick the City", 
    options=df[df["State"].isin(state)]["City"].unique() if state else df["City"].unique(), 
    help="Filter data by city."
)
# Apply the location filters
filtered_df = filter_by_location(df, region, state, city)

# Tab 3: Visualizations
with tab3:
    st.header("Visualisations")
    col1, col2 = st.columns(2)
    # Category-wise data handling in the first column
    with col1:
        st.subheader("Category-wise Data Visualization")
        # Expander for category chart type selection
        category_chart_type = st.selectbox(
                "Choose chart type for Category-wise data:",
                options=["Bar Chart", "Pie Chart", "Line Chart"]
            )
        # Call the generic chart function
        fig = create_chart(df, category_chart_type, "Category", "Sales", "Category-wise Sales")
        st.plotly_chart(fig)
        # View Data Option for category-wise sales
        with st.expander("View Category-wise Data"):
            st.dataframe(filtered_df[['Category', 'Sales']])  # Show a preview of the data
        # Download Region-wise Sales Data
        create_download_button(filtered_df, ['Category', 'Sales'], "Category_Sales")
    # Region-wise data handling in the second column
    with col2:
        st.subheader("Region-wise Data Visualization")
        # Expander for region chart type selection
        region_chart_type = st.selectbox(
                "Choose chart type for Region-wise data:",
                options=["Bar Chart", "Pie Chart", "Line Chart"]
            )
        # Call the generic chart function
        fig = create_chart(df, region_chart_type, "Region", "Sales", "Region-wise Sales")
        st.plotly_chart(fig)
        # View Data Option for region-wise sales
        with st.expander("View Region-wise Data"):
            st.dataframe(filtered_df[['Region', 'Sales']])  # Show a preview of the data
        # Download option for region-wise sales data
        create_download_button(filtered_df, ['Region', 'Sales'], "Region_Sales")
    # Time Series Analysis Visualization
    st.subheader("Time Series Analysis")
    fig = plot_time_series_sales(filtered_df)  # Plot time series sales
    st.plotly_chart(fig, use_container_width=True, key="time_series_chart")  # Display chart
    # Generate insights and display on dashboard
    insights = generate_sales_insights(filtered_df)
    # Displaying the insights in the dashboard
    st.write(f"Total Sales: ${insights['total_sales']:.2f}")
    st.write(f"Best Performing Month: {insights['best_month']} with ${insights['best_month_value']:.2f} in sales.")
    st.write(f"Worst Performing Month: {insights['worst_month']} with ${insights['worst_month_value']:.2f} in sales.")
    st.write(f"Latest Month-over-Month Growth: {insights['latest_growth']:.2f}%")
    # View Data Option for time series sales
    with st.expander("View Time Series Data"):
        st.dataframe(filtered_df[['month_year', 'Sales']])  # Show a preview of the data
    # Download option for time series sales data
    create_download_button(filtered_df, ['month_year', 'Sales'], "Time_Series_Sales")
    # Treemap Visualization for hierarchical sales view
    st.subheader("Hierarchical View of Sales")
    fig = plot_sales_treemap(filtered_df)  # Plot sales treemap
    st.plotly_chart(fig, use_container_width=True, key="treemap_sales_chart")  # Display chart
    # View Data Option for treemap sales
    with st.expander("View Treemap Data"):
        st.dataframe(filtered_df[['Region', 'Category', 'Sub-Category', 'Sales']])  # Show a preview of the data
    # Download option for treemap data
    create_download_button(filtered_df, ['Region', 'Category', 'Sub-Category', 'Sales'], "Treemap_Sales")
    # Scatter Plot for relationship between sales and profit
    st.subheader("Relationship Between Sales and Profit")
    fig = plot_scatter_sales_profit(filtered_df)  # Plot scatter plot
    st.plotly_chart(fig, use_container_width=True, key="scatter_sales_profit_chart")  # Display chart
    # Generate insights from sales and profit data grouped by category
    insights = generate_sales_profit_insights_by_category(filtered_df)
    # Displaying insights for each category
    for index, row in insights["category_grouped"].iterrows():
        st.write(f"### Category: {row['Category']}")
        st.write(f"**Total Sales**: ${row['total_sales']:.2f}")
        st.write(f"**Total Profit**: ${row['total_profit']:.2f}")
        st.write(f"**Average Sales to Profit Ratio**: {row['avg_sales_profit_ratio']:.2f}")
        st.write(f"**Sales-Profit Correlation**: {row['correlation']:.2f}")
        col1, col2, col3 = st.columns([1, 2, 2])  # Create 3 columns
        with col1:
            # Display the top 5 products by sales and profit for the category
            st.write(f"#### Top 5 Products by Sales:")
            category_sales = insights["top_sales_by_category"][insights["top_sales_by_category"]["Category"] == row["Category"]]
            st.write(category_sales[['Sub-Category', 'Sales']])
        with col2:
            st.write(f"#### Top 5 Products by Profit:")
            category_profit = insights["top_profit_by_category"][insights["top_profit_by_category"]["Category"] == row["Category"]]
            st.write(category_profit[['Sub-Category', 'Profit']])
        with col3:
            # Display outliers for the category
            st.write(f"#### Outliers (High Sales, Low Profit):")
            category_outliers = insights["outliers_by_category"][insights["outliers_by_category"]["Category"] == row["Category"]]
            st.write(category_outliers[['Sub-Category', 'Sales', 'Profit']])
    # Download option for scatter plot data
    create_download_button(filtered_df, ['Sales', 'Profit'], "Sales_Profit_Data")
    # Viewing first 500 rows of filtered data
    st.subheader("First 500 Rows of Filtered Data")
    first_500_rows = filtered_df.head(500)  # Extract first 500 rows for preview
    st.dataframe(first_500_rows)  # Display first 500 rows
    # Download button for the first 500 rows
    create_download_button(first_500_rows, first_500_rows.columns, "First_500_Rows")
    # --- Month-wise Sub-Category Sales Summary ---
    st.subheader(":point_right: Month-wise Sub-Category Sales Summary")
    with st.expander("Summary_Table"):
        df_sample = df[0:5][["Region", "State", "City", "Category", "Sales", "Profit", "Quantity"]]  # Sample data for table
        fig = ff.create_table(df_sample, colorscale="Cividis")  # Create table visualization
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Month-wise Sub-Category Table")
        filtered_df["month"] = filtered_df["Order Date"].dt.month_name()  # Extract month names from order date
        sub_category_Year = pd.pivot_table(data=filtered_df, values="Sales", index=["Sub-Category"], columns="month")  # Pivot table for sales by sub-category and month
        st.write(sub_category_Year.style.background_gradient(cmap="Blues"))  # Display the pivot table with color gradient

# Tab 4: Forecasting
with tab4:
    # Visualization Filters for user interaction
    st.subheader("Filters for Visualization")
    col1, col2 = st.columns(2)  # Create two columns for filters
    with col1:
        region_filter = st.selectbox("Select Region", options=df['Region'].unique())  # Region filter
    with col2:
        category_filter = st.selectbox("Select Category", options=df['Category'].unique())  # Category filter
    # Filter the data based on user input
    filtered_data = df[(df['Region'] == region_filter) & (df['Category'] == category_filter)]  # Apply the region and category filters
    st.write(f"Displaying data for Region: {region_filter} and Category: {category_filter}")  # Display filter info
    st.dataframe(filtered_data.head(100))  # Display the filtered data
    # Data Visualizations for filtered data
    st.subheader("Visualizations")
    if df is not None:
        create_charts(filtered_data)  # Call the function to create visualizations for the filtered data
    # Forecasting Section for future sales prediction
    st.subheader("Future Sales Prediction")
    if df is not None:
        forecast_period = st.slider("Select forecast period (months)", min_value=1, max_value=24, value=12)  # Select forecast period
        forecast_sales(filtered_data, forecast_period)  # Predict future sales based on selected forecast period
    # Download Section for processed and filtered data
    st.download_button("Download Processed Data", df.to_csv(index=False), file_name="processed_superstore.csv", mime="text/csv")  # Download processed data
    st.download_button("Download Filtered Data", filtered_data.to_csv(index=False), file_name="filtered_superstore.csv", mime="text/csv")  # Download filtered data




