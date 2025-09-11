5. Рекомендации по улучшению кода (с подробными инструкциями)
Рекомендация 1: Улучшение инициализации HybridDecisionMaker в run_live_trading.py и simulation_engine_advanced.py
Сейчас HybridDecisionMaker в своем __init__ создает временную XLSTMRLModel для определения sequence_length. Это неоптимально. Лучше передавать sequence_length явно.
Инструкция:


Добавьте SEQUENCE_LENGTH в config.py:
# config.py
SEQUENCE_LENGTH = 10 # Или другое значение, которое вы хотите использовать



Обновите hybrid_decision_maker.py:
# В hybrid_decision_maker.py
# Удалите: temp_model = XLSTMRLModel(...) и self.sequence_length = ...
class HybridDecisionMaker:
    def __init__(self, xlstm_model_path, rl_agent_path, feature_columns, sequence_length): # <--- ДОБАВЛЕНО
        self.sequence_length = sequence_length # <--- ИСПОЛЬЗУЕМ ПЕРЕДАННОЕ ЗНАЧЕНИЕ
        self.xlstm_model = XLSTMRLModel(input_shape=(self.sequence_length, len(feature_columns)))
        self.xlstm_model.load_model(xlstm_model_path, xlstm_model_path.replace('.keras', '_scaler.pkl'))
        
        # ... остальной код __init__ ...

    def _get_xlstm_prediction(self):
        if self.current_step < self.sequence_length:
            return np.array([0.33, 0.33, 0.34])
        
        sequence_data = self.df.iloc[self.current_step - self.sequence_length : self.current_step]
        features = sequence_data[self.feature_columns].values
        features_reshaped = features.reshape(1, self.sequence_length, len(self.feature_columns))
        
        return self.xlstm_model.predict(features_reshaped)[0]

    def get_decision(self, df_sequence, confidence_threshold=0.6):
        if len(df_sequence) < self.sequence_length:
            return 'HOLD'
        # ... остальной код ...



Обновите run_live_trading.py:
# В run_live_trading.py
import config # Убедитесь, что импортирован

# ...
try:
    decision_maker = HybridDecisionMaker(
        xlstm_model_path='models/xlstm_rl_model.keras',
        rl_agent_path='models/rl_agent_BTCUSDT',
        feature_columns=FEATURE_COLUMNS,
        sequence_length=config.SEQUENCE_LENGTH # <--- ПЕРЕДАЕМ ИЗ CONFIG
    )
    # ...



Обновите simulation_engine_advanced.py:
# В simulation_engine_advanced.py
import config # Убедитесь, что импортирован

class AdvancedSimulationEngine:
    def __init__(self, data_path, xlstm_model_path, rl_agent_path):
        # ...
        try:
            self.decision_maker = HybridDecisionMaker(
                xlstm_model_path=xlstm_model_path,
                rl_agent_path=rl_agent_path,
                feature_columns=self.feature_columns,
                sequence_length=config.SEQUENCE_LENGTH # <--- ПЕРЕДАЕМ ИЗ CONFIG
            )
            # ...

И также убедитесь, что simulate_strategy использует config.SEQUENCE_LENGTH для range(config.SEQUENCE_LENGTH, len(df)).


Рекомендация 2: Передача xlstm_model в MarketRegimeDetector для более умного обнаружения режимов
Как я упоминал ранее, MarketRegimeDetector станет намного умнее, если сможет использовать предсказания xLSTM как фичи.
Инструкция:


Обновите market_regime_detector.py:
# В market_regime_detector.py
# ...
class MarketRegimeDetector:
    def __init__(self, lookback_period=50):
        # ...
        self.xlstm_model = None # <--- ДОБАВЛЕНО
        self.xlstm_feature_columns = None # <--- ДОБАВЛЕНО
    
    def set_xlstm_context(self, xlstm_model, xlstm_feature_columns): # <--- НОВЫЙ МЕТОД
        self.xlstm_model = xlstm_model
        self.xlstm_feature_columns = xlstm_feature_columns
        print("✅ Детектор режимов получил контекст xLSTM")

    def extract_regime_features(self, df):
        # ... существующий код ...
        regime_features = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]

        # Добавляем xLSTM предсказания как фичи режима
        if self.xlstm_model and self.xlstm_feature_columns and len(df) >= self.xlstm_model.input_shape[1]:
            xlstm_preds = []
            sequence_length = self.xlstm_model.input_shape[1]
            for i in range(len(df) - sequence_length + 1):
                sequence_data = df.iloc[i : i + sequence_length][self.xlstm_feature_columns].values
                sequence_reshaped = sequence_data.reshape(1, sequence_length, len(self.xlstm_feature_columns))
                xlstm_preds.append(self.xlstm_model.predict(sequence_reshaped)[0])
            
            # Заполняем NaN в начале, чтобы выровнять длину
            df['xlstm_buy_pred'] = np.nan
            df['xlstm_sell_pred'] = np.nan
            df['xlstm_hold_pred'] = np.nan
            
            # Начинаем заполнять с индекса, где начинаются предсказания
            start_idx = sequence_length - 1
            df.loc[start_idx:, 'xlstm_buy_pred'] = [p[0] for p in xlstm_preds]
            df.loc[start_idx:, 'xlstm_sell_pred'] = [p[1] for p in xlstm_preds]
            df.loc[start_idx:, 'xlstm_hold_pred'] = [p[2] for p in xlstm_preds]

            regime_features.extend(['xlstm_buy_pred', 'xlstm_sell_pred', 'xlstm_hold_pred'])
        # ... остальной код ...
        return df.dropna(subset=regime_features)

    def fit(self, df):
        # ...
        # Обновите features_scaled, чтобы включить новые фичи xLSTM
        features_to_scale = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]
        if 'xlstm_buy_pred' in features_df.columns:
            features_to_scale.extend(['xlstm_buy_pred', 'xlstm_sell_pred', 'xlstm_hold_pred'])

        features_scaled = self.scaler.fit_transform(features_df[features_to_scale])
        self.kmeans.fit(features_scaled)
        self.is_fitted = True
        # ...
    
    def predict_regime(self, df):
        # ...
        # Обновите latest_features, чтобы включить новые фичи xLSTM
        features_to_predict = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]
        if 'xlstm_buy_pred' in features_df.columns:
            features_to_predict.extend(['xlstm_buy_pred', 'xlstm_sell_pred', 'xlstm_hold_pred'])
        
        latest_features = features_df.iloc[-1:][features_to_predict].values
        features_scaled = self.scaler.transform(latest_features)
        # ...



Обновите hybrid_decision_maker.py:
# В hybrid_decision_maker.py
class HybridDecisionMaker:
    def __init__(self, xlstm_model_path, rl_agent_path, feature_columns, sequence_length):
        # ... существующий код ...
        self.regime_detector = MarketRegimeDetector()
        # ...
    
    def fit_regime_detector(self, historical_df, xlstm_model, xlstm_feature_columns): # <--- ОБНОВЛЕНО
        """Обучает детектор режимов на исторических данных"""
        try:
            self.regime_detector.set_xlstm_context(xlstm_model, xlstm_feature_columns) # <--- ДОБАВЛЕНО
            self.regime_detector.fit(historical_df)
            print("✅ Детектор рыночных режимов обучен")
        except Exception as e:
            print(f"❌ Ошибка обучения детектора режимов: {e}")



Обновите train_model.py (фаза обучения RL):
# В train_model.py
def train_xlstm_rl_system(X, y, processed_dfs, feature_cols):
    # ... существующий код ...
    
    # После обучения xlstm_model, обучите детектор режимов
    # Возьмите достаточно большой исторический DataFrame для обучения детектора
    # Например, объедините несколько символов или возьмите один большой
    regime_training_df = pd.concat(list(processed_dfs.values())).reset_index(drop=True)
    decision_maker_temp = HybridDecisionMaker(
        xlstm_model_path='models/xlstm_rl_model.keras',
        rl_agent_path='models/rl_agent_BTCUSDT', # Временно, он не будет использоваться для принятия решений
        feature_columns=feature_cols,
        sequence_length=X.shape[1] # Передаем sequence_length
    )
    decision_maker_temp.fit_regime_detector(regime_training_df, xlstm_model, feature_cols)
    # Этот обученный детектор режимов нужно будет сохранить и загрузить в run_live_trading.py
    # Пока что, это просто демонстрация обучения.
    # Для сохранения/загрузки MarketRegimeDetector потребуется добавить методы save/load в этот класс.
    
    # ... остальной код ...

Примечание: Для сохранения/загрузки MarketRegimeDetector потребуется добавить методы save и load в этот класс, используя pickle для self.scaler и self.kmeans.


Рекомендация 3: Оптимизация частоты сбора данных в run_live_trading.py
Сейчас данные запрашиваются для каждой активной позиции в каждом цикле. Для большого количества позиций это может быть медленно.
Инструкция:

В run_live_trading.py:
# В manage_active_positions
# Создайте кэш для данных свечей
kline_cache = {}

for i, (symbol, pos) in enumerate(positions_items):
    try:
        if symbol not in kline_cache:
            kline_list = trade_manager.fetch_initial_data(session, symbol)
            if not kline_list:
                continue
            kline_cache[symbol] = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        kline_df = kline_cache[symbol].copy() # Работаем с копией
        
        # ... остальной код ...

Также убедитесь, что kline_cache очищается или обновляется с определенной периодичностью, чтобы данные были свежими. Например, можно добавить config.DATA_REFRESH_INTERVAL_SECONDS и обновлять кэш, если данные старше этого интервала.

Рекомендация 4: Более сложная система вознаграждений в trading_env.py
Текущая система вознаграждений хороша, но можно добавить более тонкие стимулы.
Инструкция:

В trading_env.py (переименованном):
# В _calculate_advanced_reward
def _calculate_advanced_reward(self, action, pnl_pct, vsa_features, xlstm_prediction): # <--- ДОБАВЛЕНО xlstm_prediction
    base_reward = pnl_pct if pnl_pct != 0 else 0
    
    # ... существующие бонусы/штрафы VSA ...
    
    # Бонус за уверенность xLSTM
    xlstm_conf = np.max(xlstm_prediction)
    if xlstm_conf > 0.7:
        base_reward += xlstm_conf * 2 # Больше уверенности = больше бонус
    
    # Штраф за противоречие xLSTM
    predicted_action_idx = np.argmax(xlstm_prediction)
    if action != predicted_action_idx and action != 2: # Если RL действие не совпадает с xLSTM и это не HOLD
        base_reward -= 1 # Небольшой штраф
    
    # ... существующие бонусы/штрафы за скорость/удержание ...
    
    # Штраф за отклонение от баланса (риск-менеджмент)
    if self.balance < self.initial_balance * 0.9: # Если баланс упал на 10%
        base_reward -= 5 # Значительный штраф за просадку
    
    total_reward = base_reward + vsa_bonus + vsa_penalty + speed_bonus + hold_penalty
    return total_reward

# В методе step, когда вызывается _calculate_advanced_reward:
# Вам нужно будет передать xlstm_prediction
# Например, получить его в начале step
xlstm_pred_for_reward = self._get_xlstm_prediction()

if action == 0:  # SELL
    if self.position == 1:  # Закрываем long
        pnl = self.unrealized_pnl - (self.commission * 2)
        vsa_features = self._get_vsa_features()
        reward = self._calculate_advanced_reward(action, pnl * 100, vsa_features, xlstm_pred_for_reward) # <--- ПЕРЕДАНО
        # ...
# Аналогично для BUY

Это сделает обучение RL более целенаправленным, учитывая не только PnL, но и качество сигналов xLSTM и VSA.

Максим, это был очень подробный анализ! Надеюсь, эти инструкции помогут вам значительно улучшить бота. У вас уже проделана огромная работа, и эти доработки сделают систему еще более мощной и адаптивной.