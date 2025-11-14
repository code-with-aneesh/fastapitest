from fastapi import FastAPI, Query, HTTPException, Depends
from pydantic import BaseModel
import yfinance as yf
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- NEW: SQLAlchemy Database Setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./stock_cache.db"
# check_same_thread=False is needed only for SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- NEW: Caching Configuration ---
CACHE_DURATION = timedelta(minutes=15)

# --- Pydantic Models ---
# (These are unchanged)
class StockInfo(BaseModel):
    """Basic company information."""
    ticker: str
    companyName: str
    sector: str
    industry: str
    marketCap: int
    summary: str

class HistoricalData(BaseModel):
    """A single day's historical data point."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

class PredictionResponse(BaseModel):
    """Response model for the prediction endpoint."""
    ticker: str
    last_close: float
    predicted_next_close: float
    prediction_model: str = "Simple Linear Regression (Last 90 days)"

# --- NEW: Database Models for Caching ---
class CachedInfo(Base):
    __tablename__ = "cached_info"
    ticker = Column(String, primary_key=True, index=True)
    data_json = Column(JSON)
    last_fetched = Column(DateTime)

class CachedHistorical(Base):
    __tablename__ = "cached_historical"
    cache_key = Column(String, primary_key=True, index=True) # e.g., "AAPL-3mo"
    data_json = Column(JSON)
    last_fetched = Column(DateTime)

class CachedPrediction(Base):
    __tablename__ = "cached_prediction"
    ticker = Column(String, primary_key=True, index=True)
    data_json = Column(JSON)
    last_fetched = Column(DateTime)


# --- FastAPI Application ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="Stock-Info API",
    description="A simple API to fetch stock data and simple predictions.",
    version="1.0.0"
)

# --- NEW: Create database tables on startup ---
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/", response_class=FileResponse)
def serve_frontend():
    return "static/index.html"

# --- NEW: Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Helper Function for yfinance ---
# (Unchanged)
def get_stock_ticker(ticker: str):
    """Helper to fetch a yfinance Ticker object and handle errors."""
    stock = yf.Ticker(ticker)
    if not stock.info or stock.info.get('regularMarketPrice') is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Invalid ticker symbol or no data found: {ticker}"
        )
    return stock

# --- API Endpoints (MODIFIED for Caching) ---

@app.get("/info", response_model=StockInfo)
def get_company_info(
    ticker: str = Query(..., description="Stock ticker symbol"),
    db: Session = Depends(get_db) # --- MODIFIED ---
):
    """
    Get general information for a given stock ticker.
    Checks cache first.
    """
    # --- NEW: Caching Logic ---
    cache_record = db.query(CachedInfo).filter(CachedInfo.ticker == ticker).first()
    
    # Check if cache exists and is fresh
    if cache_record and cache_record.last_fetched > datetime.utcnow() - CACHE_DURATION:
        return StockInfo(**cache_record.data_json) # Return from cache
    
    # --- Cache miss: Fetch data ---
    try:
        stock = get_stock_ticker(ticker)
        info = stock.info
        
        return_data = StockInfo(
            ticker=ticker,
            companyName=info.get('longName', 'N/A'),
            sector=info.get('sector', 'N/A'),
            industry=info.get('industry', 'N/A'),
            marketCap=info.get('marketCap', 0),
            summary=info.get('longBusinessSummary', 'N/A')
        )
        
        # --- NEW: Save to cache ---
        db_record = CachedInfo(
            ticker=ticker, 
            data_json=return_data.model_dump(mode='json'), 
            last_fetched=datetime.utcnow()
        )
        db.merge(db_record) # merge() updates if exists, inserts if not
        db.commit()
        
        return return_data
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical", response_model=List[HistoricalData])
def get_historical_data(
    ticker: str = Query(..., description="Stock ticker symbol"),
    period: str = Query("3mo", description="Data period (e.g., 1d, 5d, 1mo, 3mo, 1y)"),
    db: Session = Depends(get_db) # --- MODIFIED ---
):
    """
    Get historical OHLCV (Open, High, Low, Close, Volume) data.
    Checks cache first.
    """
    # --- NEW: Caching Logic ---
    cache_key = f"{ticker}-{period}" # Unique key for this combination
    cache_record = db.query(CachedHistorical).filter(CachedHistorical.cache_key == cache_key).first()

    if cache_record and cache_record.last_fetched > datetime.utcnow() - CACHE_DURATION:
        # Re-create Pydantic models from the cached list of dicts
        return [HistoricalData(**item) for item in cache_record.data_json]

    # --- Cache miss: Fetch data ---
    stock = get_stock_ticker(ticker)
    data = stock.history(period=period)
    
    if data.empty:
        raise HTTPException(status_code=404, detail="No historical data found for this period.")

    data = data.reset_index()
    response_data = [
        HistoricalData(
            date=row['Date'].date(),
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=row['Volume']
        )
        for _, row in data.iterrows()
    ]
    
    # --- NEW: Save to cache ---
    # Convert list of Pydantic models to a list of dicts for JSON storage
    json_to_cache = [item.model_dump(mode='json') for item in response_data]
    db_record = CachedHistorical(
        cache_key=cache_key,
        data_json=json_to_cache,
        last_fetched=datetime.utcnow()
    )
    db.merge(db_record)
    db.commit()
    
    return response_data

@app.get("/predict", response_model=PredictionResponse)
def predict_stock(
    ticker: str = Query(..., description="Stock ticker symbol"),
    db: Session = Depends(get_db) # --- MODIFIED ---
):
    """
    Predicts the next closing price using a simple linear regression.
    Checks cache first.
    """
    # --- NEW: Caching Logic ---
    cache_record = db.query(CachedPrediction).filter(CachedPrediction.ticker == ticker).first()

    if cache_record and cache_record.last_fetched > datetime.utcnow() - CACHE_DURATION:
        return PredictionResponse(**cache_record.data_json)

    # --- Cache miss: Fetch data and predict ---
    stock = get_stock_ticker(ticker)
    data = stock.history(period="90d")

    if data.empty or len(data) < 10:
        raise HTTPException(
            status_code=400, 
            detail="Not enough historical data to make a prediction."
        )

    # --- Simple Linear Regression ---
    df = data.reset_index()
    X = np.arange(len(df)).reshape(-1, 1) 
    y = df['Close'].values
    coeffs = np.polyfit(X.flatten(), y, 1)
    
    next_day_number = len(df)
    predicted_price = (coeffs[0] * next_day_number) + coeffs[1]
    
    return_data = PredictionResponse(
        ticker=ticker,
        last_close=y[-1],
        predicted_next_close=round(predicted_price, 2)
    )
    
    # --- NEW: Save to cache ---
    db_record = CachedPrediction(
        ticker=ticker,
        data_json=return_data.model_dump(mode='json'),
        last_fetched=datetime.utcnow()
    )
    db.merge(db_record)
    db.commit()
    
    return return_data
