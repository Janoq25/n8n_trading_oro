# Gold Price Trading AI Advisor

An intelligent system for gold trading recommendation and price prediction using Deep Learning (Transformer & LSTM) and Streamlit.

## 📂 Project Structure

```
/
├── app/                  # Frontend Application
│   ├── app.py           # Main Streamlit application
│   └── .streamlit/      # Streamlit configuration
├── api/                  # Backend API Services
│   ├── servidor.py      # FastAPI server (LSTM model)
│   ├── server_transformer.py # FastAPI server (Transformer model)
│   └── train_model.py   # Script to train models via API
├── models/               # Machine Learning Models
│   ├── *.h5             # Saved Keras models
│   └── *.pkl            # Scalers and metadata
├── data/                 # Data Storage
│   ├── orohistorico.json # Historical gold price data
│   └── schema.sql       # Database schema
├── n8n/                  # Automation
│   └── workflow.json    # n8n workflow export
└── tests/                # Test Suites
    ├── test_servidor.py
    └── test_transformer_v2.py
```

## 🚀 Setup & Installation

1.  **Clone the repository**
2.  **Create a virtual environment** (recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**
    ```bash
    pip install streamlit fastapi uvicorn pandas numpy scikit-learn tensorflow plotly requests supabase reportlab
    ```

## 🏃‍♂️ How to Run

### 1. Start the Backend API
You can run either the LSTM or Transformer model server.

**Option A: Transformer Model (Recommended)**
```bash
cd api
python server_transformer.py
```
*Server will start at `http://localhost:8000`*

**Option B: LSTM Model**
```bash
cd api
python servidor.py
```

### 2. Start the Frontend App
Open a new terminal:
```bash
cd app
streamlit run app.py
```
*App will open in your browser at `http://localhost:8501`*

## 🧪 Running Tests

To run the tests, ensure you are in the project root directory.

**Test Transformer Model Logic:**
```bash
python tests/test_transformer_v2.py
```

**Test API Integration:**
1. Start the server (see above).
2. Run the test script:
```bash
python tests/test_servidor.py
```

## 🧠 Models

The project includes two Deep Learning architectures:
- **Transformer**: Uses Multi-Head Attention to capture long-term dependencies in price movements.
- **LSTM**: Uses Long Short-Term Memory networks for time-series forecasting.

Models are automatically saved to the `models/` directory after training.
