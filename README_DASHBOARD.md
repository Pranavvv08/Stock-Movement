# Stock Movement Prediction Dashboard 📈

An interactive Streamlit dashboard for visualizing and predicting stock movements using BERT embeddings and deep learning models (LSTM, GRU, Bidirectional).

## 🚀 Features

### 1. **Home Page** 🏠
- Project overview and architecture
- Technology stack information
- System workflow diagram
- Quick dataset statistics

### 2. **Data Exploration** 📈
- Interactive stock price charts with zoom and pan
- Tweet samples with sentiment analysis
- Label distribution visualization
- Statistical summaries
- Feature correlation heatmaps
- Date range filtering

### 3. **Model Performance** 🤖
- Three model architectures comparison:
  - LSTM (Baseline)
  - LSTM + GRU (Hybrid)
  - Bidirectional LSTM + GRU (Best)
- Training history visualization (accuracy/loss curves)
- Performance metrics (accuracy, precision, recall, F1-score)
- Model comparison charts

### 4. **Live Prediction** 🔮
- Custom tweet input
- Stock price context selection
- Model selection (choose from 3 models)
- Real-time prediction with confidence scores
- Visual prediction result display
- Sample tweet templates

### 5. **Insights & Conclusion** 💡
- Key findings from the analysis
- Model comparison summary
- Potential applications
- Limitations and considerations
- Future enhancement suggestions

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- At least 4GB RAM (for loading models)

## 🔧 Installation

1. **Clone the repository** (if not already done):
```bash
git clone https://github.com/Pranavvv08/Stock-Movement.git
cd Stock-Movement
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: The installation may take several minutes as it includes TensorFlow, PyTorch, and Sentence Transformers.

### Installation Tips

- **For TensorFlow 1.14.0**: This older version may require Python 3.6-3.7. Consider upgrading to TensorFlow 2.x if you encounter compatibility issues.
- **For PyTorch CPU**: The requirements.txt includes CPU-only PyTorch for better compatibility.
- **For Windows**: Use `python` instead of `python3` in commands.

## 🎯 Running the Dashboard

### Option 1: Using the batch file (Windows)
```bash
run_dashboard.bat
```

### Option 2: Using command line
```bash
streamlit run app.py
```

### Option 3: Using Python
```bash
python -m streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## 📊 Data Files Required

The dashboard expects the following files in the `Dataset/` directory:
- `AAPL.csv` - Apple stock historical data with columns: Date, Open, High, Low, Close, Adj Close, Volume, Label
- `tweets.csv` - Financial tweets with columns: Tweets, Label

The following files should be in the `model/` directory:
- `bert.npy` - Pre-computed BERT embeddings (optional, will be generated if missing)
- `lstm_weights.hdf5` - LSTM model weights
- `propose_weights.hdf5` - LSTM+GRU model weights
- `extension_weights.hdf5` - Bidirectional LSTM+GRU model weights
- `lstm_history.pckl` - LSTM training history
- `propose_history.pckl` - LSTM+GRU training history
- `extension_history.pckl` - Bidirectional training history

## 🎨 Dashboard Pages

### Navigation
Use the sidebar to navigate between different pages:
- 🏠 **Home**: Overview and introduction
- 📈 **Data Exploration**: Visualize datasets
- 🤖 **Model Performance**: Compare model metrics
- 🔮 **Live Prediction**: Make predictions on custom inputs
- 💡 **Insights & Conclusion**: Key findings and takeaways

### Interactive Features

#### Stock Price Visualization
- **Zoom**: Click and drag on the chart
- **Pan**: Hold shift and drag
- **Reset**: Double-click on the chart
- **Hover**: View detailed information

#### Date Range Selection
- Use the date pickers to filter data
- View statistics for specific time periods

#### Live Predictions
1. Enter a tweet about Apple stock
2. Select a date for stock price context
3. Choose a prediction model
4. Click "Predict Stock Movement"
5. View the prediction result with confidence score

## 🛠️ Troubleshooting

### Common Issues

**1. Module not found errors**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**2. TensorFlow compatibility issues**
```bash
# Try upgrading to TensorFlow 2.x
pip install tensorflow>=2.10.0
```

**3. BERT model download issues**
```bash
# The first run will download the BERT model (~500MB)
# Ensure you have internet connection
```

**4. Port already in use**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

**5. Models not loading**
- Ensure you've run the training notebook (`StockMovement.ipynb`) first
- Check that model files exist in the `model/` directory

### Performance Tips

- **First load is slow**: The dashboard caches data and models after the first load
- **Memory issues**: Close other applications if you run out of RAM
- **Slow predictions**: The BERT encoding may take a few seconds for the first prediction

## 📸 Screenshots

### Home Page
*(Screenshot placeholder - The home page shows project overview and architecture)*

### Data Exploration
*(Screenshot placeholder - Interactive stock price charts and tweet analysis)*

### Model Performance
*(Screenshot placeholder - Training history and metrics comparison)*

### Live Prediction
*(Screenshot placeholder - Real-time prediction interface)*

## 🎓 Educational Value

This dashboard is excellent for learning:
- **Multi-modal Deep Learning**: Combining text and numerical data
- **LSTM/GRU Architectures**: Sequential data processing
- **BERT Embeddings**: Transfer learning for NLP
- **Streamlit Development**: Interactive web applications
- **Data Visualization**: Plotly charts and graphs
- **Model Deployment**: From notebook to application

## ⚠️ Disclaimer

**IMPORTANT**: This dashboard and its predictions are for **educational purposes only**. 

- Do NOT use this as the sole basis for investment decisions
- Stock markets are complex and unpredictable
- Always consult with qualified financial professionals
- Past performance does not guarantee future results
- The creators are not responsible for any financial losses

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more stock symbols
- Implement real-time data streaming
- Add more technical indicators
- Improve model architectures
- Add more visualizations

## 📝 License

This project is for educational purposes. Please check the repository for license information.

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review the code comments in `app.py`

## 🙏 Acknowledgments

- **Streamlit**: For the amazing dashboard framework
- **Plotly**: For interactive visualizations
- **Hugging Face**: For Sentence Transformers and BERT models
- **TensorFlow/Keras**: For deep learning capabilities
- **Financial community**: For open financial data

---

**Happy Exploring! 📊📈**

Remember: This is a learning tool, not a trading system. Use responsibly and continue learning!
