from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import uvicorn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pickle
import os
import hashlib
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Predicción del Precio del Oro (Transformer)",
    description="Sistema de predicción usando Transformer con ML best practices",
    version="2.0.0"
)

# ==================== Configuración ====================

MODEL_WEIGHTS_PATH = "../models/model_transformer.weights.h5"
SCALER_PATH = "../models/scaler_transformer.pkl"
METADATA_PATH = "../models/model_metadata.pkl"

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
        for date_str, data in v.items():
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Fecha inválida: {date_str}. Use formato YYYY-MM-DD")
            if 'USD' not in data:
                raise ValueError(f"Falta el campo 'USD' para la fecha {date_str}")
            if data['USD'] <= 0:
                raise ValueError(f"El precio debe ser positivo para la fecha {date_str}")
        return v

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
    sequence_length: int
    model_loaded_from_disk: bool

class RetrainResponse(BaseModel):
    message: str
    metrics: MetricsResponse

# ==================== Clase del Modelo de Predicción (Transformer) ====================

class GoldPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 20  # Ventana de 20 días (ajustado para APIs con límite de 30 días)
        self.buy_sell_threshold = 1.5  # Umbral en % para señales BUY/SELL
        self.model_loaded_from_disk = False
        self.metrics = {
            'mae': 0.0,
            'rmse': 0.0,
            'r2_score': 0.0,
            'model_trained': False,
            'training_samples': 0,
            'mean_price': 0.0  # Para cálculo de confianza
        }
        
        # Intentar cargar modelo existente
        self._load_model_if_exists()
    
    def _calculate_data_hash(self, prices: np.array) -> str:
        """Calcula hash de los datos para detectar cambios"""
        return hashlib.md5(prices.tobytes()).hexdigest()
    
    def _load_model_if_exists(self):
        """Carga modelo y scaler desde disco si existen"""
        try:
            if os.path.exists(MODEL_WEIGHTS_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(METADATA_PATH):
                logger.info("Cargando modelo desde disco...")
                
                # Cargar metadata primero para obtener la configuración
                with open(METADATA_PATH, 'rb') as f:
                    self.metrics = pickle.load(f)
                
                # Recrear la arquitectura del modelo
                self.model = self.create_model((self.sequence_length, 1))
                
                # Cargar los pesos
                self.model.load_weights(MODEL_WEIGHTS_PATH)
                
                # Cargar scaler
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.model_loaded_from_disk = True
                logger.info("✓ Modelo cargado exitosamente desde disco")
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo: {e}")
            self.model_loaded_from_disk = False
    
    def _save_model(self):
        """Guarda pesos del modelo, scaler y metadata en disco"""
        try:
            # Guardar solo los pesos (más confiable con custom layers)
            self.model.save_weights(MODEL_WEIGHTS_PATH)
            
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(METADATA_PATH, 'wb') as f:
                pickle.dump(self.metrics, f)
            
            logger.info("✓ Modelo guardado exitosamente en disco")
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {e}")
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        Bloque Transformer Encoder con Multi-Head Attention.
        Arquitectura optimizada para secuencias de 30 días.
        """
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def create_model(self, input_shape):
        """
        Crea un modelo Transformer para predicción de series temporales.
        Arquitectura flexible para permitir futuras entradas multivariables.
        """
        inputs = keras.Input(shape=input_shape)
        
        # Transformer Blocks - Ajustados para secuencias de 30 días
        x = self.transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.15)
        x = self.transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.15)
        x = self.transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.15)
        
        # Global Average Pooling to flatten
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output layers con regularización
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def prepare_sequences(self, data, sequence_length):
        """
        Prepara secuencias para entrenamiento del modelo.
        Cada secuencia X contiene sequence_length días, y predice el día siguiente.
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def train_model(self, prices: np.array, force_retrain: bool = False):
        """
        Entrena el modelo Transformer con los datos históricos.
        
        IMPORTANTE: Evita data leakage al:
        1. Ajustar el scaler solo con datos de entrenamiento
        2. Transformar validación con el scaler ya ajustado
        """
        min_required = self.sequence_length + 10
        if len(prices) < min_required:
            raise ValueError(f"Se necesitan al menos {min_required} días de datos históricos (sequence_length={self.sequence_length})")
        
        # Si ya hay modelo y no se fuerza reentrenamiento, no hacer nada
        if self.model is not None and not force_retrain:
            logger.info("Modelo ya entrenado. Use force_retrain=True para reentrenar.")
            return
        
        logger.info(f"Entrenando modelo con {len(prices)} días de datos...")
        
        # Preparar secuencias ANTES de escalar (para dividir correctamente)
        X_full, y_full = self.prepare_sequences(prices.reshape(-1, 1), self.sequence_length)
        
        # Dividir en entrenamiento y validación (80/20)
        split_idx = int(len(X_full) * 0.8)
        X_train_raw, X_val_raw = X_full[:split_idx], X_full[split_idx:]
        y_train_raw, y_val_raw = y_full[:split_idx], y_full[split_idx:]
        
        # FIX DATA LEAKAGE: Ajustar scaler SOLO con datos de entrenamiento
        # Concatenar todas las secuencias de entrenamiento para ajustar el scaler
        train_data_for_scaler = X_train_raw.reshape(-1, 1)
        train_labels_for_scaler = y_train_raw.reshape(-1, 1)
        all_train_data = np.vstack([train_data_for_scaler, train_labels_for_scaler])
        
        self.scaler.fit(all_train_data)
        
        # Transformar datos de entrenamiento y validación
        X_train = np.array([self.scaler.transform(seq) for seq in X_train_raw])
        y_train = self.scaler.transform(y_train_raw)
        
        X_val = np.array([self.scaler.transform(seq) for seq in X_val_raw])
        y_val = self.scaler.transform(y_val_raw)
        
        # Crear y entrenar modelo
        self.model = self.create_model((self.sequence_length, 1))
        
        # Callbacks para entrenamiento
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        logger.info("Iniciando entrenamiento...")
        self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Calcular métricas en validación
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_rescaled = self.scaler.inverse_transform(y_pred)
        y_val_rescaled = self.scaler.inverse_transform(y_val)
        
        self.metrics['mae'] = float(mean_absolute_error(y_val_rescaled, y_pred_rescaled))
        self.metrics['rmse'] = float(np.sqrt(mean_squared_error(y_val_rescaled, y_pred_rescaled)))
        self.metrics['r2_score'] = float(r2_score(y_val_rescaled, y_pred_rescaled))
        self.metrics['model_trained'] = True
        self.metrics['training_samples'] = len(X_train)
        self.metrics['mean_price'] = float(np.mean(prices))
        
        logger.info(f"✓ Entrenamiento completado - MAE: {self.metrics['mae']:.2f}, RMSE: {self.metrics['rmse']:.2f}, R²: {self.metrics['r2_score']:.3f}")
        
        # Guardar modelo en disco
        self._save_model()
    
    def predict_future(self, prices: np.array, horizon_days: int) -> List[float]:
        """
        Predice precios futuros para el horizonte especificado.
        Usa predicción autoregresiva: cada predicción se usa como input para la siguiente.
        """
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Llame a train_model() primero.")
        
        # Preparar última secuencia
        last_sequence = prices[-self.sequence_length:].reshape(-1, 1)
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        # Predicción iterativa (autoregresiva)
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
        Calcula la volatilidad histórica como desviación estándar de retornos logarítmicos.
        """
        if len(prices) < 2:
            return 0.0
        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns))
    
    def calculate_confidence(self, prices: np.array, predicted_price: float) -> float:
        """
        Calcula la confianza de la predicción basada en el error del modelo.
        
        NOTA: Esta es una medida HEURÍSTICA, no una probabilidad estadística rigurosa.
        Se basa en el MAE relativo al precio promedio.
        
        Fórmula: confidence = 1 - (MAE / mean_price)
        Ajustada por volatilidad para reflejar incertidumbre del mercado.
        """
        if not self.metrics['model_trained']:
            return 0.5  # Confianza neutral si no hay modelo
        
        mean_price = self.metrics.get('mean_price', np.mean(prices))
        mae = self.metrics['mae']
        
        # Confianza base: 1 - (error relativo)
        # Si MAE es 10 y precio promedio es 2000, error relativo = 0.005 (0.5%)
        relative_error = mae / mean_price if mean_price > 0 else 1.0
        base_confidence = 1.0 - relative_error
        
        # Ajustar por volatilidad (mayor volatilidad = menor confianza)
        volatility = self.calculate_volatility(prices)
        volatility_penalty = min(0.3, volatility * 5)  # Máximo 30% de penalización
        
        confidence = base_confidence * (1 - volatility_penalty)
        
        # Limitar entre 0.3 y 0.95
        return float(np.clip(confidence, 0.3, 0.95))
    
    def generate_signal(self, price_change_percent: float, volatility: float) -> str:
        """
        Genera señal de trading BUY/SELL/HOLD.
        
        NOTA EDUCATIVA: Esta es una regla simplificada para demostración.
        En producción real, las señales deberían considerar:
        - Análisis técnico adicional
        - Gestión de riesgo
        - Contexto de mercado
        - Estrategia de trading específica
        
        Regla actual:
        - BUY: cambio > threshold% (ajustado por volatilidad)
        - SELL: cambio < -threshold%
        - HOLD: entre -threshold% y +threshold%
        """
        # Ajustar umbral según volatilidad
        # Mayor volatilidad = umbral más alto (más conservador)
        adjusted_threshold = self.buy_sell_threshold * (1 + volatility * 2)
        
        if price_change_percent > adjusted_threshold:
            return "BUY"
        elif price_change_percent < -adjusted_threshold:
            return "SELL"
        else:
            return "HOLD"

# ==================== Instancia Global del Predictor ====================

predictor = GoldPricePredictor()

# ==================== Endpoints ====================

@app.get("/")
def read_root():
    return {
        "message": "API de Predicción del Precio del Oro (Transformer)",
        "version": "2.0.0",
        "model": "Transformer with Multi-Head Attention",
        "sequence_length": predictor.sequence_length,
        "endpoints": {
            "POST /predict": "Realiza una predicción del precio del oro",
            "GET /model/metrics": "Obtiene las métricas del modelo",
            "POST /retrain": "Fuerza el reentrenamiento del modelo",
            "GET /health": "Verifica el estado de salud de la API"
        }
    }

@app.post("/predict", response_model=List[PredictionResponse])
def predict_gold_price(data: GoldDataInput):
    """
    Predice el precio futuro del oro basado en datos históricos.
    
    El modelo se entrena automáticamente en la primera llamada si no existe.
    Para reentrenar, use el endpoint /retrain.
    """
    try:
        # Extraer y ordenar datos
        sorted_dates = sorted(data.gold_data.keys())
        prices = np.array([data.gold_data[date]['USD'] for date in sorted_dates])
        
        min_required = predictor.sequence_length + 10
        if len(prices) < min_required:
            raise HTTPException(
                status_code=400,
                detail=f"Se requieren al menos {min_required} días de datos históricos (sequence_length={predictor.sequence_length})"
            )
        
        # Entrenar modelo si no existe (solo primera vez)
        if predictor.model is None:
            logger.info("Modelo no encontrado. Entrenando por primera vez...")
            predictor.train_model(prices, force_retrain=True)
        
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
        
        # Determinar señal de inversión (con lógica mejorada)
        prediction_signal = predictor.generate_signal(price_change_percent, volatility)
        
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
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/model/metrics", response_model=MetricsResponse)
def get_model_metrics():
    """
    Devuelve las métricas actuales del modelo entrenado.
    """
    return MetricsResponse(
        **predictor.metrics,
        sequence_length=predictor.sequence_length,
        model_loaded_from_disk=predictor.model_loaded_from_disk
    )

@app.post("/retrain", response_model=RetrainResponse)
def retrain_model(data: GoldDataInput):
    """
    Fuerza el reentrenamiento del modelo con nuevos datos.
    
    Use este endpoint cuando:
    - Tenga datos significativamente nuevos
    - El rendimiento del modelo haya degradado
    - Quiera actualizar el modelo con datos recientes
    """
    try:
        sorted_dates = sorted(data.gold_data.keys())
        prices = np.array([data.gold_data[date]['USD'] for date in sorted_dates])
        
        logger.info("Iniciando reentrenamiento forzado del modelo...")
        predictor.train_model(prices, force_retrain=True)
        
        metrics = MetricsResponse(
            **predictor.metrics,
            sequence_length=predictor.sequence_length,
            model_loaded_from_disk=False
        )
        
        return RetrainResponse(
            message="Modelo reentrenado exitosamente",
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Error en reentrenamiento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en el reentrenamiento: {str(e)}")

@app.get("/health")
def health_check():
    """
    Verifica el estado de salud de la API.
    """
    return {
        "status": "healthy",
        "model_trained": predictor.metrics['model_trained'],
        "model_loaded_from_disk": predictor.model_loaded_from_disk,
        "sequence_length": predictor.sequence_length,
        "timestamp": datetime.now().isoformat()
    }

# ==================== Ejecución ====================

if __name__ == "__main__":
    print("🚀 Iniciando servidor de predicción del precio del oro (Transformer v2.0)...")
    print(f"📊 Modelo: Transformer (Multi-Head Attention)")
    print(f"📏 Sequence Length: {predictor.sequence_length} días")
    print(f"💾 Model Weights Path: {MODEL_WEIGHTS_PATH}")
    print(f"🌐 Documentación disponible en: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
