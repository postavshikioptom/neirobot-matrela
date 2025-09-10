"""
LIME анализ для объяснения решений DQN в торговом боте
"""
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import os
import warnings
import sys
warnings.filterwarnings('ignore')

# Попытка импортировать необходимые библиотеки для работы с моделью
try:
    from stable_baselines3 import DQN
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Stable-baselines3 не доступен. Будет использована заглушка для модели.")

def check_required_files():
    """
    Проверка наличия необходимых файлов для анализа
    """
    print("Проверка наличия необходимых файлов:")
    required_files = [
        'trade_log.csv'
    ]
    
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            all_files_exist = False
    
    return all_files_exist

# Загрузка обученной модели DQN
def load_dqn_model(model_path='dqn_trading_model.zip'):
    """
    Загрузка обученной модели DQN
    """
    if not STABLE_BASELINES_AVAILABLE:
        print("Stable-baselines3 не доступен. Возвращается заглушка.")
        return None
        
    try:
        if os.path.exists(model_path):
            model = DQN.load(model_path)
            print(f"Модель успешно загружена из {model_path}")
            return model
        else:
            print(f"Файл модели {model_path} не найден")
            return None
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None

# Загрузка данных из trader_log.csv
def load_trading_data(log_path='trade_log.csv'):
    """
    Загрузка данных торговли из CSV файла
    """
    try:
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            print(f"Загружено {len(df)} записей из {log_path}")
            return df
        else:
            print(f"Файл данных {log_path} не найден")
            return None
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

# Подготовка данных для LIME
def prepare_data_for_lime(df):
    """
    Подготовка данных для анализа LIME с учетом структуры trade_log.csv
    """
    if df is None or len(df) == 0:
        print("Нет данных для подготовки")
        return None, None, None
    
    # Вывод информации о структуре данных
    print(f"Структура данных: {df.shape}")
    print("Столбцы:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Определение столбцов с признаками на основе структуры trade_log.csv
    # Исключаем служебные столбцы, которые точно не являются признаками
    exclude_columns = [
        'timestamp', 'symbol', 'dqn_decision', 'order_type', 'status',
        'bybit_order_id', 'error_message'
    ]
    
    # Определяем столбцы с признаками, как в train_model.py
    feature_columns = [
        'price', 'quantity', 'usdt_amount', 'pnl',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'OBV', 'ATRr_14',
        'WILLR_14', 'RSI_14', 'CCI_20_0.015', 'ADX_14', 'DMP_14', 'DMN_14',
        'xgboost_prediction',
        'kalman_price', 'kalman_trend',
        'gpr_prediction', 'gpr_confidence',
        'lstm_prediction', 'lstm_confidence'
    ]
    
    # Получаем все столбцы из DataFrame
    all_cols = df.columns.tolist()
    
    # Исключаем служебные столбцы
    for col in exclude_columns:
        if col in all_cols:
            all_cols.remove(col)
            
    # Используем все оставшиеся столбцы как признаки
    feature_columns = all_cols
    
    print(f"\nПотенциальные столбцы с признаками ({len(feature_columns)}):")
    for col in feature_columns:
        print(f"  - {col}")
    
    # Проверка на наличие числовых данных
    numeric_columns = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            print(f"  Столбец '{col}' не является числовым и будет пропущен")
    
    feature_columns = numeric_columns
    
    # Обработка пропущенных значений
    # Заменяем 'N/A' на NaN
    df.replace('N/A', np.nan, inplace=True)
    
    # Удаление строк с пропущенными значениями в столбцах признаков
    df_clean = df.copy()
    for col in feature_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean[feature_columns] = df_clean[feature_columns].fillna(0)
    
    if len(df_clean) == 0:
        print("Нет строк без пропущенных значений")
        return None, None, None
    
    # Разделение на признаки и действия DQN
    X = df_clean[feature_columns]
    
    # Создаем фиктивный столбец dqn_action, если его нет
    if 'dqn_action' not in df_clean.columns:
        print("Столбец 'dqn_action' не найден, будет создан фиктивный на основе 'dqn_decision'")
        # Преобразуем dqn_decision в числовые значения
        decision_mapping = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
        df_clean['dqn_action'] = df_clean['dqn_decision'].map(decision_mapping).fillna(0)
    
    y = df_clean['dqn_action']
    
    print(f"\nПодготовлено {len(X)} образцов для анализа")
    print(f"Число признаков: {len(feature_columns)}")
    
    return X, y, feature_columns

# Создание объяснителя LIME
def create_lime_explainer(X, feature_names):
    """
    Создание LIME объяснителя для табличных данных
    """
    if X is None or len(X) == 0:
        print("Нет данных для создания объяснителя")
        return None
        
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=feature_names,
            class_names=['Hold', 'Buy', 'Sell'],  # Предполагаемые классы действий
            mode='classification',
            discretize_continuous=True
        )
        print("LIME объяснитель успешно создан")
        return explainer
    except Exception as e:
        print(f"Ошибка при создании объяснителя: {e}")
        return None

# Заглушка для модели, если stable-baselines3 недоступен
class DummyModel:
    def __init__(self):
        pass
    
    def predict(self, X, deterministic=True):
        # Возвращает случайные действия для демонстрации
        if len(X.shape) == 1:
            # Если передан один образец
            action = np.random.choice([0, 1, 2])  # Hold, Buy, Sell
            return action, None
        else:
            # Если передана batch матрица
            actions = np.random.choice([0, 1, 2], size=X.shape[0])  # Hold, Buy, Sell
            return actions, None

# Генерация объяснений для отдельных решений
def generate_explanations(model, explainer, X, num_samples=10):
    """
    Генерация LIME объяснений для случайных решений
    """
    if model is None or explainer is None or X is None:
        print("Не хватает необходимых компонентов для генерации объяснений")
        return []
        
    explanations = []
    
    # Выбор случайных образцов для объяснения
    sample_indices = np.random.choice(X.index, size=min(num_samples, len(X)), replace=False)
    
    print(f"Генерация объяснений для {len(sample_indices)} образцов...")
    
    for i, idx in enumerate(sample_indices):
        # Получение предсказания модели для текущего образца
        sample = X.loc[idx].values.reshape(1, -1)
        
        # Получение действия от модели
        try:
            action, _ = model.predict(sample, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action[0]  # Извлекаем первое действие, если возвращается массив
        except Exception as e:
            print(f"Ошибка при получении предсказания для индекса {idx}: {e}")
            # Используем случайное действие как запасной вариант
            action = np.random.choice([0, 1, 2])
            
        # Преобразуем действие в читаемый формат
        action_names = ['Hold', 'Buy', 'Sell']
        action_name = action_names[action] if 0 <= action < len(action_names) else f"Unknown({action})"
        
        print(f"  Образец {i+1}/{len(sample_indices)} (индекс {idx}): действие = {action_name}")
        
        # Генерация объяснения с помощью LIME
        try:
            # Для функции предсказания используем обертку, которая возвращает вероятности
            def predict_fn(x):
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                
                # Паддинг до 34 признаков
                if x.shape[1] < 34:
                    padding = np.zeros((x.shape[0], 34 - x.shape[1]))
                    x = np.hstack([x, padding])
                
                # Получаем предсказание от модели
                action, _ = model.predict(x, deterministic=True)
                
                # Преобразуем в вероятности
                probs = np.zeros((x.shape[0], 3))
                for i, act in enumerate(action):
                    probs[i, act] = 1
                return probs
            
            exp = explainer.explain_instance(
                sample[0],
                predict_fn,
                num_features=10,
                top_labels=1
            )
            explanations.append((idx, action_name, exp))
        except Exception as e:
            print(f"Ошибка при генерации объяснения для индекса {idx}: {e}")
    
    return explanations

# Сохранение объяснений в файл
def save_explanations(explanations, output_path='lime_explanations.txt'):
    """
    Сохранение объяснений в текстовый файл
    """
    if not explanations:
        print("Нет объяснений для сохранения")
        return
        
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("LIME Объяснения для решений DQN\n")
            f.write("=" * 40 + "\n\n")
            
            for idx, action_name, exp in explanations:
                f.write(f"Объяснение для решения с индексом {idx} (действие: {action_name}):\n")
                exp_list = exp.as_list(label=exp.available_labels()[0])
                for feature, weight in exp_list:
                    f.write(f"  {feature}: {weight:.4f}\n")
                f.write("\n")
        print(f"Объяснения сохранены в {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении объяснений: {e}")

# Создание HTML отчета с объяснениями
def create_html_report(explanations, output_path='lime_explanations.html'):
    """
    Создание HTML отчета с объяснениями
    """
    if not explanations:
        print("Нет объяснений для создания отчета")
        return
        
    try:
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>LIME Объяснения для решений DQN</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .explanation { border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }
        .explanation h2 { color: #555; margin-top: 0; }
        .feature { margin: 5px 0; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <h1>LIME Объяснения для решений DQN</h1>
"""
        
        for idx, action_name, exp in explanations:
            html_content += f"""
    <div class="explanation">
        <h2>Решение с индексом {idx} (действие: {action_name})</h2>
"""
            
            exp_list = exp.as_list(label=exp.available_labels()[0])
            for feature, weight in exp_list:
                # Определяем класс для стилизации
                weight_class = "positive" if weight > 0 else "negative"
                html_content += f'        <div class="feature {weight_class}">{feature}: {weight:.4f}</div>\n'
            
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML отчет сохранен в {output_path}")
    except Exception as e:
        print(f"Ошибка при создании HTML отчета: {e}")

# Проверка результатов анализа
def check_analysis_results():
    """
    Проверка результатов LIME анализа
    """
    print("\nПроверка результатов LIME анализа:")
    result_files = [
        'lime_explanations.txt',
        'lime_explanations.html'
    ]
    
    results_found = True
    for file in result_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            results_found = False
    
    return results_found

# Основная функция для запуска всего анализа
def run_full_lime_analysis():
    """
    Основная функция для выполнения полного LIME анализа
    """
    print("LIME анализ для объяснения решений DQN")
    print("=" * 40)
    
    # Проверка наличия необходимых файлов
    if not check_required_files():
        print("\n❌ Некоторые необходимые файлы отсутствуют.")
        return False
    
    # Загрузка данных
    print("\n1. Загрузка данных...")
    df = load_trading_data()
    if df is None:
        print("Не удалось загрузить данные. Завершение работы.")
        return False
    
    # Загрузка модели
    print("\n2. Загрузка модели...")
    if STABLE_BASELINES_AVAILABLE:
        model = load_dqn_model()
        if model is None:
            print("Не удалось загрузить модель. Будет использована заглушка.")
            model = DummyModel()
    else:
        print("Stable-baselines3 не доступен. Будет использована заглушка.")
        model = DummyModel()
    
    # Подготовка данных
    print("\n3. Подготовка данных для LIME...")
    X, y, feature_names = prepare_data_for_lime(df)
    if X is None or len(X) == 0:
        print("Нет данных для анализа. Завершение работы.")
        return False
    
    # Создание объяснителя
    print("\n4. Создание LIME объяснителя...")
    explainer = create_lime_explainer(X, feature_names)
    if explainer is None:
        print("Не удалось создать объяснитель. Завершение работы.")
        return False
    
    # Генерация объяснений
    print("\n5. Генерация объяснений...")
    explanations = generate_explanations(model, explainer, X, num_samples=5)
    
    if len(explanations) == 0:
        print("Не удалось сгенерировать объяснения. Завершение работы.")
        return False
    
    # Сохранение объяснений
    print("\n6. Сохранение результатов...")
    save_explanations(explanations)
    create_html_report(explanations)
    
    print(f"\nСгенерировано {len(explanations)} объяснений")
    
    # Вывод примера объяснения
    if explanations:
        idx, action_name, exp = explanations[0]
        print(f"\nПример объяснения для решения с индексом {idx} (действие: {action_name}):")
        exp_list = exp.as_list(label=exp.available_labels()[0])
        for feature, weight in exp_list[:5]:  # Показываем первые 5 признаков
            print(f"  {feature}: {weight:.4f}")
    
    # Проверка результатов
    if check_analysis_results():
        print("\n✅ LIME анализ успешно завершен и результаты сохранены.")
        return True
    else:
        print("\n⚠️  LIME анализ выполнен, но результаты не найдены.")
        return True

# Основная функция
def main():
    """
    Основная функция для выполнения LIME анализа
    """
    # Если скрипт запущен напрямую, выполняем полный анализ
    if __name__ == "__main__":
        run_full_lime_analysis()

if __name__ == "__main__":
    run_full_lime_analysis()