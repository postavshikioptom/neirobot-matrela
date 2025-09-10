import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os

class MarketRegimeDetector:
    """
    Детектор рыночных режимов для адаптации стратегии
    """
    
    def __init__(self, lookback_period=50):
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  # n_init to avoid warning
        self.is_fitted = False
        self.xlstm_model = None
        self.xlstm_feature_columns = None
        
        # Названия режимов
        self.regime_names = {
            0: 'TRENDING_UP',
            1: 'TRENDING_DOWN', 
            2: 'SIDEWAYS_HIGH_VOL',
            3: 'SIDEWAYS_LOW_VOL'
        }

    def set_xlstm_context(self, xlstm_model, xlstm_feature_columns):
        self.xlstm_model = xlstm_model
        self.xlstm_feature_columns = xlstm_feature_columns
        print("✅ Детектор режимов получил контекст xLSTM")
    
    def extract_regime_features(self, df):
        """Извлекает признаки для определения режима рынка"""
        
        # Ценовые признаки
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['trend_strength'] = df['close'].rolling(20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0)
        
        # Объемные признаки
        df['volume_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
        df['volume_volatility'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        
        # Технические признаки
        df['rsi_regime'] = np.where(df['RSI_14'] > 70, 1, np.where(df['RSI_14'] < 30, -1, 0))
        df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        # VSA режимные признаки
        df['vsa_activity'] = df['vsa_strength'].rolling(10).std()
        df['vsa_direction'] = df['vsa_strength'].rolling(10).mean()
        
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
        
        return df.dropna(subset=regime_features)
    
    def fit(self, df):
        """Обучает детектор на исторических данных"""
        
        features_df = self.extract_regime_features(df)
        if len(features_df) < self.lookback_period:
            raise ValueError("Недостаточно данных для обучения детектора режимов")
        
        # Нормализация и кластеризация
        features_to_scale = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]
        if 'xlstm_buy_pred' in features_df.columns:
            features_to_scale.extend(['xlstm_buy_pred', 'xlstm_sell_pred', 'xlstm_hold_pred'])

        features_scaled = self.scaler.fit_transform(features_df[features_to_scale])
        self.kmeans.fit(features_scaled)
        self.is_fitted = True
        
        # Анализируем характеристики кластеров
        labels = self.kmeans.labels_
        self.analyze_clusters(features_df, labels)
        
        print("✅ Детектор рыночных режимов обучен")
        return self
    
    def analyze_clusters(self, features_df, labels):
        """Анализирует характеристики найденных кластеров"""
        
        print("\n📊 АНАЛИЗ РЫНОЧНЫХ РЕЖИМОВ:")
        print("-" * 50)
        
        for cluster in range(4):
            cluster_data = features_df[labels == cluster]
            if len(cluster_data) > 0:
                avg_volatility = cluster_data['volatility'].mean()
                avg_trend = cluster_data['trend_strength'].mean()
                avg_vsa_activity = cluster_data['vsa_activity'].mean()
                
                print(f"\nРежим {cluster} ({self.regime_names.get(cluster, 'UNKNOWN')}):")
                print(f"  - Волатильность: {avg_volatility:.4f}")
                print(f"  - Сила тренда: {avg_trend:.4f}")
                print(f"  - VSA активность: {avg_vsa_activity:.4f}")
                print(f"  - Количество периодов: {len(cluster_data)}")
    
    def predict_regime(self, df):
        """Предсказывает текущий рыночный режим"""
        
        if not self.is_fitted:
            raise ValueError("Детектор должен быть обучен перед предсказанием")
        
        features_df = self.extract_regime_features(df)
        if len(features_df) == 0:
            return 'UNKNOWN', 0.0
        
        # Берем последние признаки
        features_to_predict = [
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]
        if 'xlstm_buy_pred' in features_df.columns:
            features_to_predict.extend(['xlstm_buy_pred', 'xlstm_sell_pred', 'xlstm_hold_pred'])
        
        latest_features = features_df.iloc[-1:][features_to_predict].values
        features_scaled = self.scaler.transform(latest_features)
        
        # Предсказываем режим
        regime_id = self.kmeans.predict(features_scaled)[0]
        regime_name = self.regime_names.get(regime_id, 'UNKNOWN')
        
        # Вычисляем уверенность (расстояние до центра кластера)
        distances = self.kmeans.transform(features_scaled)[0]
        confidence = 1.0 - (distances[regime_id] / distances.sum()) if distances.sum() > 0 else 0
        
        return regime_name, confidence
    
    def get_regime_trading_params(self, regime_name):
        """Возвращает торговые параметры для режима"""
        
        params = {
            'TRENDING_UP': {
                'confidence_threshold': 0.5,
                'take_profit': 2.0,
                'stop_loss': -1.0,
                'position_size_multiplier': 1.2
            },
            'TRENDING_DOWN': {
                'confidence_threshold': 0.5,
                'take_profit': 2.0,
                'stop_loss': -1.0,
                'position_size_multiplier': 1.2
            },
            'SIDEWAYS_HIGH_VOL': {
                'confidence_threshold': 0.7,
                'take_profit': 1.0,
                'stop_loss': -1.5,
                'position_size_multiplier': 0.8
            },
            'SIDEWAYS_LOW_VOL': {
                'confidence_threshold': 0.8,
                'take_profit': 0.8,
                'stop_loss': -0.8,
                'position_size_multiplier': 0.6
            }
        }
        
        return params.get(regime_name, params['SIDEWAYS_LOW_VOL'])

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