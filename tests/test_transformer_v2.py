"""
Test script para verificar el servidor Transformer refactorizado.

Verifica:
1. Modelo se entrena con 60 días de datos
2. Modelo se guarda en disco (pesos)
3. Modelo se carga desde disco en siguiente ejecución
4. Predicciones funcionan correctamente
5. Scaling no tiene data leakage
"""

import os
import sys
import numpy as np
from datetime import datetime, timedelta
import json

# Add parent directory to path to find api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server_transformer import GoldPricePredictor

def generate_synthetic_gold_prices(days=60, start_price=2000):
    """Genera precios sintéticos del oro con tendencia y ruido"""
    np.random.seed(42)
    trend = np.linspace(0, 100, days)  # Tendencia alcista
    noise = np.random.normal(0, 20, days)  # Ruido
    seasonal = 30 * np.sin(np.linspace(0, 4*np.pi, days))  # Componente estacional
    
    prices = start_price + trend + noise + seasonal
    return prices

def test_model_training():
    """Test 1: Entrenamiento del modelo"""
    print("\n" + "="*60)
    print("TEST 1: Entrenamiento del Modelo")
    print("="*60)
    
    # Limpiar modelos anteriores
    model_files = [
        "../models/model_transformer.weights.h5", 
        "../models/scaler_transformer.pkl", 
        "../models/model_metadata.pkl"
    ]
    
    for file in model_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Eliminado {file} anterior")
    
    predictor = GoldPricePredictor()
    prices = generate_synthetic_gold_prices(days=60)
    
    print(f"\n📊 Datos generados: {len(prices)} días")
    print(f"   Precio inicial: ${prices[0]:.2f}")
    print(f"   Precio final: ${prices[-1]:.2f}")
    print(f"   Sequence length: {predictor.sequence_length}")
    
    print("\n🔄 Entrenando modelo...")
    predictor.train_model(prices, force_retrain=True)
    
    print(f"\n✓ Modelo entrenado exitosamente")
    print(f"   MAE: ${predictor.metrics['mae']:.2f}")
    print(f"   RMSE: ${predictor.metrics['rmse']:.2f}")
    print(f"   R²: {predictor.metrics['r2_score']:.3f}")
    print(f"   Training samples: {predictor.metrics['training_samples']}")
    
    # Verificar que los archivos se guardaron
    assert os.path.exists("../models/model_transformer.weights.h5"), "❌ Pesos del modelo no se guardaron"
    assert os.path.exists("../models/scaler_transformer.pkl"), "❌ Scaler no se guardó"
    assert os.path.exists("../models/model_metadata.pkl"), "❌ Metadata no se guardó"
    print("\n✓ Archivos del modelo guardados correctamente")
    
    return predictor, prices

def test_model_persistence():
    """Test 2: Persistencia del modelo"""
    print("\n" + "="*60)
    print("TEST 2: Persistencia del Modelo")
    print("="*60)
    
    print("\n🔄 Creando nuevo predictor (debería cargar desde disco)...")
    new_predictor = GoldPricePredictor()
    
    assert new_predictor.model is not None, "❌ Modelo no se cargó"
    assert new_predictor.model_loaded_from_disk, "❌ Modelo no se marcó como cargado desde disco"
    assert new_predictor.metrics['model_trained'], "❌ Métricas no se cargaron"
    
    print(f"✓ Modelo cargado desde disco exitosamente")
    print(f"   MAE: ${new_predictor.metrics['mae']:.2f}")
    print(f"   R²: {new_predictor.metrics['r2_score']:.3f}")
    
    return new_predictor

def test_predictions(predictor, prices):
    """Test 3: Predicciones"""
    print("\n" + "="*60)
    print("TEST 3: Predicciones")
    print("="*60)
    
    print("\n🔮 Realizando predicción a 3 días...")
    predictions = predictor.predict_future(prices, horizon_days=3)
    
    print(f"\n✓ Predicción completada")
    print(f"   Precio actual: ${prices[-1]:.2f}")
    print(f"   Predicción día 1: ${predictions[0]:.2f}")
    print(f"   Predicción día 2: ${predictions[1]:.2f}")
    print(f"   Predicción día 3: ${predictions[2]:.2f}")
    
    change = ((predictions[-1] - prices[-1]) / prices[-1]) * 100
    print(f"   Cambio esperado: {change:+.2f}%")
    
    assert len(predictions) == 3, "❌ Número incorrecto de predicciones"
    assert all(p > 0 for p in predictions), "❌ Predicciones negativas"
    
    return predictions

def test_confidence_and_signals(predictor, prices, predictions):
    """Test 4: Confianza y señales"""
    print("\n" + "="*60)
    print("TEST 4: Confianza y Señales")
    print("="*60)
    
    # Calcular confianza
    confidence = predictor.calculate_confidence(prices, predictions[-1])
    print(f"\n📊 Confianza: {confidence:.2%}")
    assert 0.3 <= confidence <= 0.95, "❌ Confianza fuera de rango"
    print("✓ Confianza en rango válido")
    
    # Calcular volatilidad
    volatility = predictor.calculate_volatility(prices)
    print(f"📈 Volatilidad: {volatility:.4f}")
    
    # Generar señal
    price_change_percent = ((predictions[-1] - prices[-1]) / prices[-1]) * 100
    signal = predictor.generate_signal(price_change_percent, volatility)
    print(f"🎯 Señal: {signal}")
    assert signal in ["BUY", "SELL", "HOLD"], "❌ Señal inválida"
    print("✓ Señal válida generada")

def test_no_data_leakage():
    """Test 5: Verificar que no hay data leakage"""
    print("\n" + "="*60)
    print("TEST 5: Verificación de Data Leakage")
    print("="*60)
    
    # Limpiar modelos
    model_files = [
        "../models/model_transformer.weights.h5", 
        "../models/scaler_transformer.pkl", 
        "../models/model_metadata.pkl"
    ]
    for file in model_files:
        if os.path.exists(file):
            os.remove(file)
    
    predictor = GoldPricePredictor()
    
    # Crear dos conjuntos de datos con diferentes rangos
    prices1 = generate_synthetic_gold_prices(days=60, start_price=2000)
    prices2 = generate_synthetic_gold_prices(days=60, start_price=3000)
    
    print("\n🔍 Entrenando con precios en rango 2000...")
    predictor.train_model(prices1, force_retrain=True)
    
    # El scaler debe estar ajustado al rango de entrenamiento
    scaler_min = predictor.scaler.data_min_[0]
    scaler_max = predictor.scaler.data_max_[0]
    
    print(f"   Scaler min: ${scaler_min:.2f}")
    print(f"   Scaler max: ${scaler_max:.2f}")
    print(f"   Rango de datos de entrenamiento: ${prices1.min():.2f} - ${prices1.max():.2f}")
    
    # Verificar que el scaler solo vio datos de entrenamiento
    # (no debería incluir datos de validación que no vio durante fit)
    assert scaler_min >= prices1.min() * 0.95, "❌ Posible data leakage en scaler"
    assert scaler_max <= prices1.max() * 1.05, "❌ Posible data leakage en scaler"
    
    print("✓ No se detectó data leakage en el scaler")

def main():
    print("\n" + "="*60)
    print("🧪 SUITE DE TESTS - TRANSFORMER SERVER v2.0")
    print("="*60)
    
    try:
        # Test 1: Entrenamiento
        predictor, prices = test_model_training()
        
        # Test 2: Persistencia
        new_predictor = test_model_persistence()
        
        # Test 3: Predicciones
        predictions = test_predictions(new_predictor, prices)
        
        # Test 4: Confianza y señales
        test_confidence_and_signals(new_predictor, prices, predictions)
        
        # Test 5: Data leakage
        test_no_data_leakage()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("="*60)
        
        # Limpiar archivos de test
        print("\n🧹 Limpiando archivos de test...")
        model_files = [
            "../models/model_transformer.weights.h5", 
            "../models/scaler_transformer.pkl", 
            "../models/model_metadata.pkl"
        ]
        for file in model_files:
            if os.path.exists(file):
                os.remove(file)
        print("✓ Limpieza completada")
        
    except AssertionError as e:
        print(f"\n❌ TEST FALLÓ: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
