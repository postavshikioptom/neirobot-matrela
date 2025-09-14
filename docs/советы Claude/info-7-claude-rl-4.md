Анализирую детально весь переписанный код по рекомендациям из файла info-7-claude-rl-3.md. Вижу отличную работу по интеграции xLSTM + VSA + RL системы! Есть несколько ошибок и предложения по улучшению:
🔍 ОБНАРУЖЕННЫЕ ОШИБКИ
1. Ошибка в xlstm_memory_cell.py
# ПРОБЛЕМА: В методе call используется tf.reduce_mean с axis=0 для батча
memory_read = tf.matmul(memory_prev, self.W_mem)
i = tf.nn.sigmoid(tf.matmul(inputs, self.W_i) + tf.matmul(h_prev, self.U_i) + 
                 tf.reduce_mean(memory_read, axis=0, keepdims=True) + self.b_i)  # ❌ ОШИБКА

# ИСПРАВЛЕНИЕ:
i = tf.nn.sigmoid(tf.matmul(inputs, self.W_i) + tf.matmul(h_prev, self.U_i) + 
                 memory_read + self.b_i)  # ✅ ПРАВИЛЬНО

2. Ошибка в trading_env.py - маппинг действий
# ПРОБЛЕМА: Несогласованность в маппинге действий RL vs xLSTM
# В коде есть: rl_to_xlstm_map = {0: 1, 1: 0, 2: 2}
# Но это создает путаницу

# ИСПРАВЛЕНИЕ - стандартизируем маппинг:
# RL actions: 0=SELL, 1=BUY, 2=HOLD  
# xLSTM output: [BUY_prob, SELL_prob, HOLD_prob] -> indices 0, 1, 2

def _calculate_advanced_reward(self, action, pnl_pct, vsa_features, xlstm_prediction):
    # ... существующий код ...
    
    # Исправленный маппинг
    predicted_action_idx = np.argmax(xlstm_prediction)
    xlstm_to_rl_map = {0: 1, 1: 0, 2: 2}  # xLSTM_BUY->RL_BUY, xLSTM_SELL->RL_SELL
    
    if action != 2 and action != xlstm_to_rl_map.get(predicted_action_idx):
        base_reward -= 1

3. Ошибка в run_live_trading.py - глобальные переменные
# ПРОБЛЕМА: performance_monitor и notification_system используются без инициализации
def run_trading_loop():
    # ❌ Переменные объявлены локально, но используются в других функциях
    performance_monitor = PerformanceMonitor()
    notification_system = NotificationSystem()

# ИСПРАВЛЕНИЕ:
# В начале файла добавить:
performance_monitor = None
notification_system = None

def run_trading_loop():
    global performance_monitor, notification_system
    performance_monitor = PerformanceMonitor()
    notification_system = NotificationSystem()

💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ
Рекомендация 1: Улучшенная инициализация HybridDecisionMaker
# config.py - ДОБАВИТЬ:
SEQUENCE_LENGTH = 10

# hybrid_decision_maker.py - ОБНОВИТЬ:
class HybridDecisionMaker:
    def __init__(self, xlstm_model_path, rl_agent_path, feature_columns, sequence_length):
        self.sequence_length = sequence_length  # ✅ Используем переданное значение
        self.xlstm_model = XLSTMRLModel(input_shape=(self.sequence_length, len(feature_columns)))
        # ... остальной код ...

# run_live_trading.py - ОБНОВИТЬ:
import config

decision_maker = HybridDecisionMaker(
    xlstm_model_path='models/xlstm_rl_model.keras',
    rl_agent_path='models/rl_agent_BTCUSDT',
    feature_columns=FEATURE_COLUMNS,
    sequence_length=config.SEQUENCE_LENGTH  # ✅ Из config
)

Рекомендация 2: Интеграция xLSTM в MarketRegimeDetector
# market_regime_detector.py - ДОБАВИТЬ:
class MarketRegimeDetector:
    def __init__(self, lookback_period=50):
        # ... существующий код ...
        self.xlstm_model = None
        self.xlstm_feature_columns = None
    
    def set_xlstm_context(self, xlstm_model, xlstm_feature_columns):
        """Устанавливает контекст xLSTM для улучшения детекции режимов"""
        self.xlstm_model = xlstm_model
        self.xlstm_feature_columns = xlstm_feature_columns
        print("✅ Детектор режимов получил контекст xLSTM")

    def extract_regime_features(self, df):
        # ... существующий код ...
        
        # Добавляем xLSTM предсказания как фичи режима
        if self.xlstm_model and len(df) >= 10:  # Минимум для sequence
            xlstm_preds = []
            for i in range(len(df) - 9):  # 10-элементные последовательности
                sequence_data = df.iloc[i:i+10][self.xlstm_feature_columns].values
                sequence_reshaped = sequence_data.reshape(1, 10, len(self.xlstm_feature_columns))
                xlstm_preds.append(self.xlstm_model.predict(sequence_reshaped)[0])
            
            # Заполняем первые 9 строк NaN
            df['xlstm_buy_pred'] = np.nan
            df['xlstm_sell_pred'] = np.nan  
            df['xlstm_hold_pred'] = np.nan
            
            # Заполняем предсказания начиная с 10-й строки
            if xlstm_preds:
                df.loc[9:, 'xlstm_buy_pred'] = [p[0] for p in xlstm_preds]
                df.loc[9:, 'xlstm_sell_pred'] = [p[1] for p in xlstm_preds]
                df.loc[9:, 'xlstm_hold_pred'] = [p[2] for p in xlstm_preds]
                
                regime_features.extend(['xlstm_buy_pred', 'xlstm_sell_pred', 'xlstm_hold_pred'])
        
        return df.dropna(subset=regime_features)

Рекомендация 3: Оптимизация частоты сбора данных
# run_live_trading.py - ОБНОВИТЬ manage_active_positions:
def manage_active_positions(session, decision_maker):
    active_positions = load_active_positions()
    if not active_positions:
        return

    print(f"Открыто сделок: {opened_trades_counter}/{OPEN_TRADE_LIMIT}. Активных позиций: {len(active_positions)}")
    
    # ✅ Кэш для данных свечей
    kline_cache = {}
    
    for symbol, pos in active_positions.items():
        try:
            # ✅ Используем кэш
            if symbol not in kline_cache:
                kline_list = trade_manager.fetch_initial_data(session, symbol)
                if not kline_list:
                    continue
                kline_cache[symbol] = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            kline_df = kline_cache[symbol].copy()
            # ... остальная логика ...

Рекомендация 4: Расширенная система вознаграждений
# trading_env.py - УЛУЧШИТЬ _calculate_advanced_reward:
def _calculate_advanced_reward(self, action, pnl_pct, vsa_features, xlstm_prediction):
    base_reward = pnl_pct if pnl_pct != 0 else 0
    
    # ✅ Существующие VSA бонусы...
    
    # ✅ НОВЫЕ УЛУЧШЕНИЯ:
    
    # Бонус за уверенность xLSTM
    xlstm_conf = np.max(xlstm_prediction)
    if xlstm_conf > 0.7:
        base_reward += xlstm_conf * 2
    
    # Штраф за противоречие xLSTM  
    predicted_action_idx = np.argmax(xlstm_prediction)
    xlstm_to_rl_map = {0: 1, 1: 0, 2: 2}  # BUY->BUY, SELL->SELL, HOLD->HOLD
    if action != 2 and action != xlstm_to_rl_map.get(predicted_action_idx):
        base_reward -= 1
    
    # ✅ Штраф за риск-менеджмент
    if self.balance < self.initial_balance * 0.9:
        base_reward -= 5
    
    # ✅ Бонус за быстрое закрытие прибыльных позиций
    if pnl_pct > 0 and self.steps_in_position < 20:
        base_reward += 2
    
    # ✅ Штраф за долгое удержание убыточных позиций
    if pnl_pct < 0 and self.steps_in_position > 30:
        base_reward -= 3
    
    return base_reward + vsa_bonus + vsa_penalty + speed_bonus + hold_penalty

Рекомендация 5: Методы сохранения/загрузки для MarketRegimeDetector
# market_regime_detector.py - ДОБАВИТЬ:
import pickle
import os

class MarketRegimeDetector:
    # ... существующий код ...
    
    def save_detector(self, path='models/market_regime_detector.pkl'):
        """Сохраняет обученный детектор режимов"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        detector_data = {
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'is_fitted': self.is_fitted,
            'lookback_period': self.lookback_period
        }
        with open(path, 'wb') as f:
            pickle.dump(detector_data, f)
        print(f"✅ Детектор режимов сохранен: {path}")
    
    def load_detector(self, path='models/market_regime_detector.pkl'):
        """Загружает обученный детектор режимов"""
        with open(path, 'rb') as f:
            detector_data = pickle.load(f)
        
        self.scaler = detector_data['scaler']
        self.kmeans = detector_data['kmeans'] 
        self.is_fitted = detector_data['is_fitted']
        self.lookback_period = detector_data['lookback_period']
        print(f"✅ Детектор режимов загружен: {path}")

Рекомендация 6: Улучшенная обработка ошибок
# run_live_trading.py - ДОБАВИТЬ:
import psutil
import gc

def monitor_system_resources():
    """Расширенный мониторинг системных ресурсов"""
    memory_percent = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent()
    
    if memory_percent > 80:
        print(f"⚠️ Высокое использование памяти: {memory_percent:.1f}%")
        gc.collect()  # Принудительная сборка мусора
        
    if cpu_percent > 90:
        print(f"⚠️ Высокая загрузка CPU: {cpu_percent:.1f}%")
        
    return {'memory': memory_percent, 'cpu': cpu_percent}

# В главном цикле добавить:
if loop_counter % 100 == 0:
    system_stats = monitor_system_resources()
    if system_stats['memory'] > 85:
        notification_system.send_system_alert(f"Критическое использование памяти: {system_stats['memory']:.1f}%")

🏆 ОБЩАЯ ОЦЕНКА
Код выглядит отлично! Вы успешно реализовали:
✅ xLSTM с настоящей расширенной памятью
✅ VSA анализ с детальными сигналами
✅ RL агента с продвинутыми наградами
✅ Гибридную систему принятия решений
✅ Детектор рыночных режимов
✅ Систему уведомлений и мониторинга
Исправив указанные ошибки и внедрив рекомендации, у вас будет мощная адаптивная торговая система нового поколения!