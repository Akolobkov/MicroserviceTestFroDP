import requests
import pandas as pd
import io
from typing import Dict, Any

# Конфигурация
BASE_URL = "http://localhost:8000"
TEST_CSV_PATH = "test_data.csv"


def create_test_csv():
    """Создание тестового CSV файла"""
    test_data = {
        'vibration_g': [0.66369, 0.16466, 0.14312, 0.33353, 0.01],
        'temperature_c': [94.16, 97.18, 98.07, 99.88, 98.36],
        'power_factor': [0.9238, 0.9298, 0.9274, 0.9062, 0.9271]
    }
    df = pd.DataFrame(test_data)
    df.to_csv(TEST_CSV_PATH, index=False)
    print(f"Тестовый CSV файл создан: {TEST_CSV_PATH}")
    return df


def test_health_check():
    """Тестирование health check эндпоинта"""
    print("\n" + "=" * 50)
    print("1. Тестирование health check")
    print("=" * 50)

    response = requests.get(f"{BASE_URL}/health")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Health check успешен")
        print(f"  Status: {data['status']}")
        print(f"  Models loaded: {data['models_loaded']}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False


def test_predict_endpoint(csv_path: str):
    """Тестирование predict эндпоинта"""
    print("\n" + "=" * 50)
    print("2. Тестирование predict эндпоинта")
    print("=" * 50)

    # Открываем и отправляем CSV файл
    with open(csv_path, 'rb') as f:
        files = {'file': (csv_path, f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/predict", files=files)

    if response.status_code == 200:
        data = response.json()

        # Парсим CSV строки обратно в DataFrame
        status_df = pd.read_csv(io.StringIO(data['machine_status']))
        ttf_df = pd.read_csv(io.StringIO(data['time_to_failure_hours']))

        print(f"✓ Predict запрос успешен")
        print(f"\n  Предсказанные статусы машины:")
        print(f"  {status_df}")
        print(f"\n  Предсказанное время до отказа (часы):")
        print(f"  {ttf_df}")

        return True, status_df, ttf_df
    else:
        print(f"✗ Predict запрос失败: {response.status_code}")
        print(f"  Response: {response.text}")
        return False, None, None


def test_invalid_file():
    """Тестирование с некорректным файлом"""
    print("\n" + "=" * 50)
    print("3. Тестирование с некорректным файлом")
    print("=" * 50)

    # Отправляем не CSV файл
    files = {'file': ('test.txt', b'not a csv file', 'text/plain')}
    response = requests.post(f"{BASE_URL}/predict", files=files)

    if response.status_code != 200:
        print(f"✓ Некорректный файл правильно отклонен: {response.status_code}")
        return True
    else:
        print(f"✗ Некорректный файл не должен был пройти")
        return False


def run_performance_test(num_requests: int = 10):
    """Тестирование производительности"""
    print("\n" + "=" * 50)
    print(f"4. Тестирование производительности ({num_requests} запросов)")
    print("=" * 50)

    import time

    with open(TEST_CSV_PATH, 'rb') as f:
        files = {'file': (TEST_CSV_PATH, f, 'text/csv')}

        times = []
        successful = 0

        for i in range(num_requests):
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/predict", files=files)
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                successful += 1
                times.append(elapsed_time)

            # Перемещаем указатель файла в начало для следующего запроса
            f.seek(0)

        if times:
            avg_time = sum(times) / len(times)
            print(f"✓ Успешных запросов: {successful}/{num_requests}")
            print(f"  Среднее время ответа: {avg_time:.3f} секунд")
            print(f"  Мин. время: {min(times):.3f} сек")
            print(f"  Макс. время: {max(times):.3f} сек")
        else:
            print("✗ Все запросы упали")


def test_with_custom_data():
    """Тестирование с кастомными данными через API"""
    print("\n" + "=" * 50)
    print("5. Тестирование с кастомными данными")
    print("=" * 50)

    # Создаем тестовые данные в памяти
    test_data = pd.DataFrame({
        'vibration_g': [0.5, 0.8, 1.2],
        'temperature_c': [95.0, 100.0, 110.0],
        'power_factor': [0.95, 0.88, 0.75]
    })

    csv_buffer = io.StringIO()
    test_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    files = {'file': ('test.csv', csv_buffer.getvalue().encode(), 'text/csv')}
    response = requests.post(f"{BASE_URL}/predict", files=files)

    if response.status_code == 200:
        data = response.json()
        status_df = pd.read_csv(io.StringIO(data['machine_status']))
        ttf_df = pd.read_csv(io.StringIO(data['time_to_failure_hours']))

        print("✓ Кастомные данные успешно обработаны")
        print(f"\n  Входные данные:")
        print(test_data)
        print(f"\n  Результат предсказаний:")
        print(pd.concat([status_df, ttf_df], axis=1))
        return True
    else:
        print(f"✗ Ошибка: {response.status_code}")
        return False


def main():
    """Основная функция тестирования"""
    print("🚀 Начало тестирования микросервиса")
    print(f"Base URL: {BASE_URL}")

    # Создаем тестовый CSV файл
    df_original = create_test_csv()

    # Запускаем тесты
    health_ok = test_health_check()

    if not health_ok:
        print("\n❌ Сервис недоступен. Убедитесь, что он запущен:")
        print("   python your_service.py")
        return

    predict_ok, status_df, ttf_df = test_predict_endpoint(TEST_CSV_PATH)

    if predict_ok:
        test_invalid_file()
        run_performance_test(num_requests=5)
        test_with_custom_data()

        # Сравнение размеров
        print("\n" + "=" * 50)
        print("6. Сравнение входных и выходных данных")
        print("=" * 50)
        print(f"Входных записей: {len(df_original)}")
        print(f"Предсказаний статуса: {len(status_df)}")
        print(f"Предсказаний TTF: {len(ttf_df)}")

        print("\n✅ Все тесты завершены успешно!")
    else:
        print("\n❌ Критические тесты не пройдены")


if __name__ == "__main__":
    main()