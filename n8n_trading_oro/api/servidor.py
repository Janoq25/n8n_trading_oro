from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import uvicorn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import json
import pickle
import os
from pathlib import Path

app = FastAPI(
    title="API de Predicción del Precio del Oro",
    description="Sistema de predicción usando LSTM mejorado",
    version="2.0.0"
)

# ==================== Configuración de Rutas ====================

MODELS_DIR = Path("../models")
MODEL_PATH = MODELS_DIR / "gold_lstm.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Crear directorio de modelos si no existe
MODELS_DIR.mkdir(exist_ok=True)

# ==================== Modelos Pydantic ====================

class GoldDataInput(BaseModel):
    gold_data: Dict[str, Dict[str, float]] = Field(..., description="Datos históricos del oro")
    timeframe: int = Field(..., ge=7, le=365, description="Ventana temporal en días")
    investment_amount: float = Field(..., gt=0, description="Monto de inversión en USD")
    prediction_horizon_days: int = Field(..., ge=1, le=3, description="Horizonte de predicción (1-3 días)")
    
    @validator('gold_data')
    def validate_gold_data(cls, v):
        if not v:
            raise ValueError("gold_data no puede estar vacío")
        
        # Detectar fechas duplicadas
        dates = list(v.keys())
        if len(dates) != len(set(dates)):
            raise ValueError("Se detectaron fechas duplicadas en gold_data")
        
        # Validar formato y valores
        for date_str, data in v.items():
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Fecha inválida: {date_str}. Use formato YYYY-MM-DD")
            if 'USD' not in data:
                raise ValueError(f"Falta el campo 'USD' para la fecha {date_str}")
            if data['USD'] <= 0:
                raise ValueError(f"El precio debe ser positivo para la fecha {date_str}")
        
        # Validar cantidad mínima de datos (sequence_length=30 + margen)
        if len(v) < 30:
            raise ValueError(f"Se requieren al menos 30 días de datos históricos. Proporcionados: {len(v)}")
        
        return v

class TrainModelInput(BaseModel):
    gold_data: Dict[str, Dict[str, float]] = Field(..., description="Datos históricos del oro para entrenamiento")
    epochs: Optional[int] = Field(50, ge=10, le=200, description="Número de épocas de entrenamiento")
    batch_size: Optional[int] = Field(16, ge=8, le=64, description="Tamaño del batch")
    
    @validator('gold_data')
    def validate_gold_data(cls, v):
        if not v:
            raise ValueError("gold_data no puede estar vacío")
        
        # Detectar fechas duplicadas
        dates = list(v.keys())
        if len(dates) != len(set(dates)):
            raise ValueError("Se detectaron fechas duplicadas en gold_data")
        
        # Validar formato y valores
        for date_str, data in v.items():
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Fecha inválida: {date_str}. Use formato YYYY-MM-DD")
            if 'USD' not in data:
                raise ValueError(f"Falta el campo 'USD' para la fecha {date_str}")
            if data['USD'] <= 0:
                raise ValueError(f"El precio debe ser positivo para la fecha {date_str}")
        
        # Validar cantidad mínima de datos
        if len(v) < 30:
            raise ValueError(f"Se requieren al menos 30 días de datos históricos para entrenar. Proporcionados: {len(v)}")
        
        return v

class TrainModelResponse(BaseModel):
    success: bool
    message: str
    mae: float
    rmse: float
    r2_score: float
    training_samples: int
    model_saved: bool

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    predicted_price: float
    current_price: float
    price_change: float
    price_change_percent: float
    horizon: str
    horizon_days: int
    volatility: float
    prediction_path: List[float]

class MetricsResponse(BaseModel):
    mae: float
    rmse: float
    r2_score: float
    model_trained: bool
    training_samples: int

# ==================== Clase del Modelo de Predicción ====================

class GoldPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 20  # Cambiado de 10 a 30 días
        self.is_trained = False
        self.model_path = MODEL_PATH
        self.scaler_path = SCALER_PATH
        self.metrics = {
            'mae': 0.0,
            'rmse': 0.0,
            'r2_score': 0.0,
            'model_trained': False,
            'training_samples': 0
        }
        
        # Intentar cargar modelo existente
        self._load_model_if_exists()
    
    def build_model(self, input_shape, conv_filters=64, lstm_units_1=64, lstm_units_2=32, 
                    dense_units=16, dropout_rate=0.2):
        """
        Crea un modelo mejorado con Conv1D + LSTM + Dense
        
        Args:
            input_shape: Forma de entrada (sequence_length, features)
            conv_filters: Número de filtros en la capa Conv1D
            lstm_units_1: Unidades en la primera capa LSTM
            lstm_units_2: Unidades en la segunda capa LSTM
            dense_units: Unidades en la capa Dense
            dropout_rate: Tasa de dropout
        """
        model = Sequential([
            # Capa convolucional para extraer características locales
            Conv1D(filters=conv_filters, kernel_size=3, activation='relu', 
                   padding='same', input_shape=input_shape),
            Dropout(dropout_rate),
            
            # Primera capa LSTM
            LSTM(lstm_units_1, return_sequences=True),
            Dropout(dropout_rate),
            
            # Segunda capa LSTM
            LSTM(lstm_units_2, return_sequences=False),
            Dropout(dropout_rate),
            
            # Capas densas
            Dense(dense_units, activation='relu'),
            Dropout(dropout_rate),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_sequences(self, data, sequence_length):
        """Prepara secuencias para entrenamiento del modelo LSTM"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def train_model(self, prices: np.array, epochs: int = 50, batch_size: int = 16):
        """
        Entrena el modelo LSTM con los datos históricos
        
        Args:
            prices: Array de precios históricos
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
        """
        if len(prices) < self.sequence_length + 5:
            raise ValueError(f"Se necesitan al menos {self.sequence_length + 5} días de datos históricos")
        
        # Normalizar datos - FIT solo con datos de entrenamiento
        prices_scaled = self.scaler.fit_transform(prices.reshape(-1, 1))
        
        # Preparar secuencias
        X, y = self.prepare_sequences(prices_scaled, self.sequence_length)
        
        # Dividir en entrenamiento y validación
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Crear modelo con arquitectura mejorada
        self.model = self.build_model((self.sequence_length, 1))
        
        # Entrenar con early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Calcular métricas
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_rescaled = self.scaler.inverse_transform(y_pred)
        y_val_rescaled = self.scaler.inverse_transform(y_val)
        
        self.metrics['mae'] = float(mean_absolute_error(y_val_rescaled, y_pred_rescaled))
        self.metrics['rmse'] = float(np.sqrt(mean_squared_error(y_val_rescaled, y_pred_rescaled)))
        self.metrics['r2_score'] = float(r2_score(y_val_rescaled, y_pred_rescaled))
        self.metrics['model_trained'] = True
        self.metrics['training_samples'] = len(X_train)
        self.is_trained = True
        
        # Guardar modelo automáticamente después de entrenar
        self.save_model()
    
    def save_model(self):
        """Guarda el modelo y el scaler en disco"""
        try:
            # Guardar modelo
            self.model.save(self.model_path)
            
            # Guardar scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"✅ Modelo guardado en {self.model_path}")
            print(f"✅ Scaler guardado en {self.scaler_path}")
            return True
        except Exception as e:
            print(f"❌ Error al guardar modelo: {str(e)}")
            return False
    
    def load_model(self):
        """Carga el modelo y el scaler desde disco"""
        try:
            # Cargar modelo
            self.model = load_model(self.model_path)
            
            # Cargar scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            self.metrics['model_trained'] = True
            
            print(f"✅ Modelo cargado desde {self.model_path}")
            print(f"✅ Scaler cargado desde {self.scaler_path}")
            return True
        except FileNotFoundError:
            print("ℹ️ No se encontraron archivos de modelo previos")
            return False
        except Exception as e:
            print(f"❌ Error al cargar modelo: {str(e)}")
            print("⚠️ El archivo puede estar corrupto. Se requiere reentrenar.")
            return False
    
    def _load_model_if_exists(self):
        """Intenta cargar el modelo al inicializar si existe"""
        if self.model_path.exists() and self.scaler_path.exists():
            self.load_model()
    
    def predict_future(self, prices: np.array, horizon_days: int) -> tuple:
        """
        Predice precios futuros para el horizonte especificado
        
        Args:
            prices: Array de precios históricos
            horizon_days: Número de días a predecir
            
        Returns:
            Lista de precios predichos
        """
        if not self.is_trained or self.model is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo primero usando /model/train")
        
        # Preparar última secuencia
        last_sequence = prices[-self.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        # Predicción iterativa
        for _ in range(horizon_days):
            current_input = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            next_pred = self.model.predict(current_input, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Actualizar secuencia con la nueva predicción
            current_sequence = np.vstack([current_sequence, next_pred.reshape(-1, 1)])
        
        # Desnormalizar predicciones
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = self.scaler.inverse_transform(predictions_array)
        
        return predictions_rescaled.flatten().tolist()
    
    def calculate_volatility(self, prices: np.array) -> float:
        """
        Calcula la volatilidad histórica usando log-returns
        
        Args:
            prices: Array de precios históricos
            
        Returns:
            Volatilidad (desviación estándar de log-returns)
        """
        # Usar log-returns en vez de returns simples
        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns))
    
    def calculate_confidence(self, prices: np.array, predicted_price: float) -> float:
        """
        Calcula la confianza de la predicción integrando MAE, RMSE y volatilidad
        
        Args:
            prices: Array de precios históricos
            predicted_price: Precio predicho
            
        Returns:
            Confianza entre 0 y 1
        """
        if not self.is_trained:
            return 0.3
        
        # Calcular volatilidad
        volatility = self.calculate_volatility(prices)
        
        # Normalizar métricas (valores más bajos = mejor)
        # Usar el precio promedio como referencia
        avg_price = np.mean(prices)
        
        # MAE normalizado (0 = perfecto, 1 = muy malo)
        mae_normalized = min(1.0, self.metrics['mae'] / (avg_price * 0.1))  # 10% del precio promedio
        
        # RMSE normalizado (0 = perfecto, 1 = muy malo)
        rmse_normalized = min(1.0, self.metrics['rmse'] / (avg_price * 0.15))  # 15% del precio promedio
        
        # R² ya está entre -inf y 1, pero lo normalizamos a [0, 1]
        r2_normalized = max(0.0, min(1.0, self.metrics['r2_score']))
        
        # Volatilidad normalizada (valores típicos de volatilidad diaria: 0.005 - 0.05)
        volatility_normalized = min(1.0, volatility / 0.05)
        
        # Combinar métricas con pesos
        # R² es positivo (mayor = mejor), los demás son negativos (menor = mejor)
        confidence = (
            0.3 * r2_normalized +                    # 30% peso a R²
            0.25 * (1 - mae_normalized) +            # 25% peso a MAE (invertido)
            0.25 * (1 - rmse_normalized) +           # 25% peso a RMSE (invertido)
            0.2 * (1 - volatility_normalized)        # 20% peso a volatilidad (invertido)
        )
        
        # Asegurar que esté en el rango [0.3, 0.95]
        return float(np.clip(confidence, 0.3, 0.95))

# ==================== Instancia Global del Predictor ====================

predictor = GoldPricePredictor()

# ==================== Endpoints ====================

@app.get("/")
def read_root():
    return {
        "message": "API de Predicción del Precio del Oro",
        "version": "2.0.0",
        "endpoints": {
            "POST /model/train": "Entrena el modelo con datos históricos",
            "POST /predict": "Realiza una predicción del precio del oro (requiere modelo entrenado)",
            "GET /model/metrics": "Obtiene las métricas del modelo",
            "GET /health": "Verifica el estado de salud de la API"
        },
        "model_status": {
            "trained": predictor.is_trained,
            "sequence_length": predictor.sequence_length
        }
    }

@app.post("/model/train", response_model=TrainModelResponse)
def train_model(data: TrainModelInput):
    """
    Entrena el modelo con datos históricos proporcionados
    """
    try:
        # Extraer y ordenar datos
        sorted_dates = sorted(data.gold_data.keys())
        prices = np.array([data.gold_data[date]['USD'] for date in sorted_dates])
        
        print(f"🎯 Iniciando entrenamiento con {len(prices)} días de datos...")
        
        # Entrenar modelo
        predictor.train_model(prices, epochs=data.epochs, batch_size=data.batch_size)
        
        print(f"✅ Entrenamiento completado")
        print(f"📊 MAE: {predictor.metrics['mae']:.2f}")
        print(f"📊 RMSE: {predictor.metrics['rmse']:.2f}")
        print(f"📊 R²: {predictor.metrics['r2_score']:.4f}")
        
        return TrainModelResponse(
            success=True,
            message="Modelo entrenado exitosamente",
            mae=predictor.metrics['mae'],
            rmse=predictor.metrics['rmse'],
            r2_score=predictor.metrics['r2_score'],
            training_samples=predictor.metrics['training_samples'],
            model_saved=True
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el entrenamiento: {str(e)}")

@app.post("/predict", response_model=List[PredictionResponse])
def predict_gold_price(data: GoldDataInput):
    """
    Predice el precio futuro del oro basado en datos históricos
    REQUIERE que el modelo esté entrenado previamente
    """
    try:
        # Validar que el modelo esté entrenado
        if not predictor.is_trained:
            raise HTTPException(
                status_code=400,
                detail="El modelo no está entrenado. Por favor, entrene el modelo primero usando POST /model/train"
            )
        
        # Extraer y ordenar datos
        sorted_dates = sorted(data.gold_data.keys())
        prices = np.array([data.gold_data[date]['USD'] for date in sorted_dates])
        
        # Validar coherencia de timeframe con datos disponibles
        if data.timeframe > len(prices):
            raise HTTPException(
                status_code=400,
                detail=f"El timeframe solicitado ({data.timeframe} días) excede la cantidad de datos disponibles ({len(prices)} días)"
            )
        
        # Validar cantidad mínima de datos
        if len(prices) < predictor.sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Se requieren al menos {predictor.sequence_length} días de datos históricos"
            )
        
        # Realizar predicción
        prediction_path = predictor.predict_future(prices, data.prediction_horizon_days)
        
        # Precio actual (último precio disponible)
        current_price = float(prices[-1])
        
        # Precio predicho (último del horizonte)
        predicted_price = float(prediction_path[-1])
        
        # Calcular cambio de precio
        price_change = predicted_price - current_price
        price_change_percent = (price_change / current_price) * 100
        
        # Calcular volatilidad
        volatility = predictor.calculate_volatility(prices)
        
        # Calcular confianza
        confidence = predictor.calculate_confidence(prices, predicted_price)
        
        # Determinar señal de inversión
        if price_change_percent > 0.5:
            prediction_signal = "BUY"
        elif price_change_percent < -0.5:
            prediction_signal = "SELL"
        else:
            prediction_signal = "HOLD"
        
        # Preparar respuesta
        horizon_text = f"{data.prediction_horizon_days} day{'s' if data.prediction_horizon_days > 1 else ''}"
        
        response = PredictionResponse(
            prediction=prediction_signal,
            confidence=confidence,
            predicted_price=predicted_price,
            current_price=current_price,
            price_change=price_change,
            price_change_percent=price_change_percent,
            horizon=horizon_text,
            horizon_days=data.prediction_horizon_days,
            volatility=volatility,
            prediction_path=prediction_path
        )
        
        return [response]
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/model/metrics", response_model=MetricsResponse)
def get_model_metrics():
    """
    Devuelve las métricas actuales del modelo entrenado
    """
    return MetricsResponse(**predictor.metrics)

@app.get("/health")
def health_check():
    """
    Verifica el estado de salud de la API
    """
    return {
        "status": "healthy",
        "model_trained": predictor.is_trained,
        "model_metrics": predictor.metrics if predictor.is_trained else None,
        "timestamp": datetime.now().isoformat()
    }

# ==================== Ejecución ====================

if __name__ == "__main__":
    print("🚀 Iniciando servidor de predicción del precio del oro...")
    print("📊 Modelo: Conv1D + LSTM (Arquitectura mejorada)")
    print(f"📏 Sequence Length: {predictor.sequence_length} días")
    print(f"🎯 Modelo entrenado: {'Sí' if predictor.is_trained else 'No'}")
    print("🌐 Documentación disponible en: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)