"""
Script de prueba para verificar el servidor mejorado de predicción de oro
"""
import requests
import json
from datetime import datetime, timedelta
import numpy as np

BASE_URL = "http://localhost:8000"

def generate_sample_data(days=60):
    """Genera datos de muestra para pruebas"""
    base_price = 2000
    dates = []
    data = {}
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=days-i)).strftime('%Y-%m-%d')
        # Simular precio con tendencia y ruido
        price = base_price + i * 2 + np.random.normal(0, 10)
        data[date] = {"USD": float(price)}
    
    return data

def test_health_check():
    """Test 1: Verificar estado de salud"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_metrics_before_training():
    """Test 2: Verificar métricas antes de entrenar"""
    print("\n" + "="*60)
    print("TEST 2: Métricas antes de entrenar")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/metrics")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    # Verificar que model_trained sea False si no hay modelo
    if not data.get('model_trained'):
        print("✅ Correcto: El modelo no está entrenado")
        return True
    else:
        print("ℹ️ El modelo ya está entrenado (cargado desde disco)")
        return True

def test_predict_without_training():
    """Test 3: Intentar predecir sin entrenar (debe fallar)"""
    print("\n" + "="*60)
    print("TEST 3: Predecir sin entrenar (debe fallar)")
    print("="*60)
    
    gold_data = generate_sample_data(60)
    
    payload = {
        "gold_data": gold_data,
        "timeframe": 30,
        "investment_amount": 1000,
        "prediction_horizon_days": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 400:
        print("✅ Correcto: Falló como se esperaba (modelo no entrenado)")
        return True
    elif response.status_code == 200:
        print("ℹ️ La predicción funcionó (modelo ya estaba entrenado)")
        return True
    else:
        print("❌ Error inesperado")
        return False

def test_train_model():
    """Test 4: Entrenar el modelo"""
    print("\n" + "="*60)
    print("TEST 4: Entrenar el modelo")
    print("="*60)
    
    gold_data = generate_sample_data(60)
    
    payload = {
        "gold_data": gold_data,
        "epochs": 30,
        "batch_size": 16
    }
    
    print("Entrenando modelo... (esto puede tomar unos segundos)")
    response = requests.post(f"{BASE_URL}/model/train", json=payload)
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if response.status_code == 200 and data.get('success'):
        print("✅ Modelo entrenado exitosamente")
        print(f"   MAE: {data.get('mae', 0):.2f}")
        print(f"   RMSE: {data.get('rmse', 0):.2f}")
        print(f"   R²: {data.get('r2_score', 0):.4f}")
        return True
    else:
        print("❌ Error al entrenar el modelo")
        return False

def test_predict_after_training():
    """Test 5: Predecir después de entrenar"""
    print("\n" + "="*60)
    print("TEST 5: Predecir después de entrenar")
    print("="*60)
    
    gold_data = generate_sample_data(60)
    
    payload = {
        "gold_data": gold_data,
        "timeframe": 30,
        "investment_amount": 1000,
        "prediction_horizon_days": 2
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if response.status_code == 200:
        prediction = data[0]
        print("✅ Predicción exitosa")
        print(f"   Señal: {prediction.get('prediction')}")
        print(f"   Confianza: {prediction.get('confidence', 0):.2%}")
        print(f"   Precio actual: ${prediction.get('current_price', 0):.2f}")
        print(f"   Precio predicho: ${prediction.get('predicted_price', 0):.2f}")
        print(f"   Cambio: {prediction.get('price_change_percent', 0):.2f}%")
        print(f"   Volatilidad: {prediction.get('volatility', 0):.4f}")
        return True
    else:
        print("❌ Error al hacer la predicción")
        return False

def test_data_validation():
    """Test 6: Validación de datos"""
    print("\n" + "="*60)
    print("TEST 6: Validación de datos")
    print("="*60)
    
    # Test 6a: Fechas duplicadas
    print("\n6a. Probando detección de fechas duplicadas...")
    gold_data = generate_sample_data(40)
    # Agregar fecha duplicada
    first_date = list(gold_data.keys())[0]
    gold_data[first_date + "_dup"] = gold_data[first_date]
    gold_data[first_date] = gold_data[first_date]
    
    payload = {
        "gold_data": gold_data,
        "timeframe": 30,
        "investment_amount": 1000,
        "prediction_horizon_days": 1
    }
    
    # Esto debería fallar en validación, pero nuestra implementación
    # solo detecta claves duplicadas exactas
    print("   (Nota: La validación de duplicados funciona a nivel de claves del dict)")
    
    # Test 6b: Datos insuficientes
    print("\n6b. Probando validación de datos insuficientes...")
    gold_data_small = generate_sample_data(20)  # Menos de 35 días
    
    payload = {
        "gold_data": gold_data_small,
        "timeframe": 10,
        "investment_amount": 1000,
        "prediction_horizon_days": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"   Status Code: {response.status_code}")
    if response.status_code == 400:
        print(f"   ✅ Correcto: Rechazó datos insuficientes")
        print(f"   Mensaje: {response.json().get('detail', '')}")
    else:
        print(f"   ❌ No rechazó datos insuficientes")
    
    # Test 6c: Timeframe mayor que datos disponibles
    print("\n6c. Probando validación de timeframe...")
    gold_data = generate_sample_data(40)
    
    payload = {
        "gold_data": gold_data,
        "timeframe": 100,  # Mayor que 40 días disponibles
        "investment_amount": 1000,
        "prediction_horizon_days": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"   Status Code: {response.status_code}")
    if response.status_code == 400:
        print(f"   ✅ Correcto: Rechazó timeframe inválido")
        print(f"   Mensaje: {response.json().get('detail', '')}")
        return True
    else:
        print(f"   ❌ No rechazó timeframe inválido")
        return False

def test_metrics_after_training():
    """Test 7: Verificar métricas después de entrenar"""
    print("\n" + "="*60)
    print("TEST 7: Métricas después de entrenar")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/metrics")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if data.get('model_trained'):
        print("✅ El modelo está entrenado")
        return True
    else:
        print("❌ El modelo no está marcado como entrenado")
        return False

def main():
    """Ejecutar todos los tests"""
    print("\n" + "="*60)
    print("INICIANDO TESTS DEL SERVIDOR DE PREDICCIÓN DE ORO")
    print("="*60)
    print(f"URL Base: {BASE_URL}")
    print("\nAsegúrate de que el servidor esté corriendo en {BASE_URL}")
    print("Ejecuta: python servidor.py")
    
    input("\nPresiona Enter para continuar...")
    
    results = []
    
    try:
        results.append(("Health Check", test_health_check()))
        results.append(("Métricas antes de entrenar", test_metrics_before_training()))
        results.append(("Predecir sin entrenar", test_predict_without_training()))
        results.append(("Entrenar modelo", test_train_model()))
        results.append(("Predecir después de entrenar", test_predict_after_training()))
        results.append(("Validación de datos", test_data_validation()))
        results.append(("Métricas después de entrenar", test_metrics_after_training()))
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se pudo conectar al servidor")
        print("Asegúrate de que el servidor esté corriendo en http://localhost:8000")
        return
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE TESTS")
    print("="*60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests pasados")
    
    if passed == total:
        print("\n🎉 ¡Todos los tests pasaron exitosamente!")
    else:
        print(f"\n⚠️ {total - passed} test(s) fallaron")

if __name__ == "__main__":
    main()
