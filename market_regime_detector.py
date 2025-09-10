import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MarketRegimeDetector:
    """
    Детектор рыночных режимов для адаптации стратегии
    """
    
    def __init__(self, lookback_period=50):
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  # n_init to avoid warning
        self.is_fitted = False
        
        # Названия режимов
        self.regime_names = {
            0: 'TRENDING_UP',
            1: 'TRENDING_DOWN', 
            2: 'SIDEWAYS_HIGH_VOL',
            3: 'SIDEWAYS_LOW_VOL'
        }
    
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
        
        return df.dropna(subset=regime_features)
    
    def fit(self, df):
        """Обучает детектор на исторических данных"""
        
        features_df = self.extract_regime_features(df)
        if len(features_df) < self.lookback_period:
            raise ValueError("Недостаточно данных для обучения детектора режимов")
        
        # Нормализация и кластеризация
        features_scaled = self.scaler.fit_transform(features_df[[
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]])
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
        latest_features = features_df.iloc[-1:][[
            'volatility', 'trend_strength', 'volume_trend', 'volume_volatility',
            'rsi_regime', 'bb_position', 'vsa_activity', 'vsa_direction'
        ]].values
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