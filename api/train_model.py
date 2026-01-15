"""
Script para entrenar el modelo usando orohistorico.json
"""
import requests
import json
import time

# Leer el archivo de datos
print("📂 Leyendo orohistorico.json...")
with open('../data/orohistorico.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Datos cargados: {len(data['gold_data'])} días de datos históricos")
print(f"   Desde: {min(data['gold_data'].keys())}")
print(f"   Hasta: {max(data['gold_data'].keys())}")

# Preparar la petición
url = "http://localhost:8000/model/train"
payload = {
    "gold_data": data["gold_data"],
    "epochs": 50,
    "batch_size": 16
}

print("\n🎯 Iniciando entrenamiento del modelo...")
print("   Epochs: 50")
print("   Batch size: 16")
print("   Esto puede tomar 1-2 minutos...\n")

start_time = time.time()

try:
    response = requests.post(url, json=payload, timeout=300)
    
    elapsed_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print("✅ ¡Entrenamiento completado exitosamente!")
        print(f"⏱️  Tiempo: {elapsed_time:.1f} segundos")
        print("\n📊 Métricas del modelo:")
        print(f"   MAE (Error Absoluto Medio): {result['mae']:.2f}")
        print(f"   RMSE (Raíz del Error Cuadrático Medio): {result['rmse']:.2f}")
        print(f"   R² Score: {result['r2_score']:.4f}")
        print(f"   Muestras de entrenamiento: {result['training_samples']}")
        print(f"   Modelo guardado: {'Sí' if result['model_saved'] else 'No'}")
        
        if result['model_saved']:
            print("\n💾 Archivos guardados:")
            print("   - models/gold_lstm.h5")
            print("   - models/scaler.pkl")
        
        print("\n✨ El modelo está listo para hacer predicciones!")
        
    else:
        print(f"❌ Error en el entrenamiento:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Respuesta: {response.text}")
        
except requests.exceptions.Timeout:
    print("⏱️ Timeout: El entrenamiento está tomando más de 5 minutos")
    print("   Esto puede indicar un problema. Verifica el servidor.")
    
except requests.exceptions.ConnectionError:
    print("❌ Error de conexión:")
    print("   No se pudo conectar al servidor en http://localhost:8000")
    print("   Asegúrate de que el servidor esté corriendo.")
    
except Exception as e:
    print(f"❌ Error inesperado: {str(e)}")
