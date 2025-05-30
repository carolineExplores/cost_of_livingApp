# Global Cost of Living Comparison App

## 📊 Description

This interactive Shiny for Python application provides visualizations for comparing the cost of living across different countries. Users can explore historical trends, compare multiple countries side-by-side, view data on a global map, analyze inflation trends, and access detailed data tables.

You can explore the interactive Shiny app by clicking the link below:

👉 [Launch the App](https://carolineexplores.shinyapps.io/livingcost_dashboard/)

## ✨ Features

* **Historical Trend Analysis:** Visualize how cost of living components and average salaries have changed over time for a selected country.
* **Country Comparisons:** Compare cost categories (housing, food, transport, etc.) and average salaries across multiple selected countries for a specific year.
* **Global Cost Map:** View an interactive world map displaying the total cost of living for various countries, with selected countries highlighted.
* **Inflation Trends:** Analyze and plot annual inflation rates for a selected country based on CPI data.
* **Detailed Data Tables:** View and download the underlying data for selected countries, including costs and salaries.

## 📁 Data Sources

The application relies on several CSV data files, which should be placed in a subfolder named `cost of living` relative to the main `app.py` file. The data is typically sourced or aligned with indicators from international organizations like the **OECD** and the **World Bank**, as well as other economic data providers.

1.  **`Housing-related-expenditure-of-households.csv`**: Contains data on household expenditure percentages across various categories (Housing, Food, Transport, etc.) for different countries. This type of data is often compiled by national statistics offices and aggregated by bodies like the OECD.
    * *Separator:* Semicolon (`;`)
    * *Encoding:* latin1
2.  **`Cost_of_living.csv`**: Provides a base total cost of living figure for various countries (likely for a specific reference year, e.g., 2025 in this app's logic). This may be based on various indices or compiled data.
    * *Separator:* Semicolon (`;`)
    * *Encoding:* utf-8
3.  **`Monthly_salary.csv`**: Contains average monthly salary data for different countries across various years. Sourced from economic data providers or national statistics.
    * *Separator:* Semicolon (`;`)
    * *Encoding:* utf-8
4.  **`CPI_2.csv`**: Contains Consumer Price Index (CPI) data for different countries across various years, used for adjusting costs and calculating inflation. This type of data is often provided by sources like the World Bank or national statistics offices.
    * *Separator:* Semicolon (`;`)
    * *Encoding:* utf-8

## ⚙️ Local Setup and Installation

To run this application locally, you'll need Python (version 3.9, 3.10, or 3.11 recommended for compatibility, similar to deployment environments like shinyapps.io).

1.  **Clone/Download the Project:**
    Ensure you have all project files, including `app.py`, the `cost of living` data subfolder, and `requirements.txt`.

2.  **Project Structure:**
    Your project directory should ideally look like this:
    ```
    your_project_folder/
    ├── app.py
    ├── cost of living/
    │   ├── CPI_2.csv
    │   ├── Cost_of_living.csv
    │   ├── Housing-related-expenditure-of-households.csv
    │   └── Monthly_salary.csv
    ├── requirements.txt
    └── venv_shinyapp/  (This will be created in the next step)
    ```

3.  **Create a Python Virtual Environment:**
    Navigate to your project's root directory in your terminal/command prompt and create a virtual environment. For example, using Python 3.9:
    ```bash
    # Replace 'py -3.9' with your Python 3.9 command if different (e.g., python3.9)
    py -3.9 -m venv venv_shinyapp
    ```

4.  **Activate the Virtual Environment:**
    * On Windows:
        ```cmd
        venv_shinyapp\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv_shinyapp/bin/activate
        ```
    Your terminal prompt should change to indicate the active environment (e.g., `(venv_shinyapp)`).

5.  **Install Dependencies:**
    Install all required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

6.  **Run the Shiny Application:**
    With the virtual environment still active, run the app:
    ```bash
    shiny run app.py --reload
    ```
    Alternatively:
    ```bash
    python -m shiny run app.py --reload
    ```
    The `--reload` flag is optional and allows the app to auto-reload when you make changes to the code.
    Open your web browser and navigate to the address shown in the terminal (usually `http://127.0.0.1:8000`).

## 🛠️ Key Technologies Used

* **Python**
* **Shiny for Python:** For the interactive web application framework.
* **Pandas:** For data manipulation and analysis.
* **Plotly:** For creating interactive charts and maps.
* **Matplotlib:** (Used as a backend or for specific plot elements if not directly visible via Plotly).
* **NumPy:** For numerical operations.

## 🚀 Deployment

This application is designed to be deployable to platforms like `shinyapps.io` or Posit Connect. Ensure you have `rsconnect-python` installed in your deployment environment and follow the platform-specific deployment instructions. Key considerations for deployment include:
* Using a compatible Python version (e.g., 3.9, 3.10, 3.11).
* Ensuring all data files are bundled correctly (e.g., by placing them in a subdirectory like `cost of living` and using relative paths in `app.py`).
* Providing an accurate `requirements.txt` file.

## 📝 Notes

* The application uses the `DATA_FOLDER = "cost of living"` setting in `app.py` to locate the data files. Ensure this matches your local data folder name.
* Country name normalization and mapping are handled within the app to reconcile differences across datasets, but the quality of this depends on the consistency of input data.
* The CPI adjustment logic in `get_cost_of_living` uses a base year of 2025 for its calculations.
* The "Debug Info" tab in the app can be helpful for inspecting loaded data shapes and selected parameters during runtime.

---
