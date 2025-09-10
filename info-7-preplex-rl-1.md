<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Архитектура модернизированного трейдинг-бота: xLSTM + VSA + RL

## Введение

На основе проведенного исследования, я собрал исчерпывающую информацию по технологиям xLSTM, VSA (Volume Spread Analysis) и Reinforcement Learning для создания комплексного руководства по архитектуре вашего нового трейдинг-бота. Эта система представляет существенный скачок от текущих двух моделей xLSTM к единой интеллектуальной системе, способной к адаптивному обучению и анализу профессиональных денег на рынке.[^1_1][^1_2][^1_3]

## Концептуальная архитектура системы

### Общая схема интеграции xLSTM + VSA + RL

Ваш новый бот будет представлять собой гибридную архитектуру с тремя взаимосвязанными компонентами:

**1. Модуль xLSTM с расширенной памятью**

- Единая модель xLSTM с mLSTM и sLSTM компонентами[^1_4][^1_5]
- Экспоненциальные ворота (exponential gating) для улучшенного контроля информации[^1_5]
- Матричная память (matrix memory) для параллелизации обучения[^1_5]
- Скалярная память с новым механизмом memory mixing[^1_5]

**2. VSA-модуль анализа профессиональных денег**

- Анализ трех ключевых параметров: объем, спред, цена закрытия[^1_2][^1_6][^1_1]
- Распознавание паттернов накопления и распределения[^1_3][^1_7]
- Детекция "умных денег" против "слабых держателей"[^1_1][^1_2]
- Сигналы силы и слабости рынка[^1_7][^1_6]

**3. RL-агент для адаптивного обучения**

- DQN, PPO или A2C алгоритмы для принятия торговых решений[^1_8][^1_9][^1_10]
- Непрерывное обучение на живых данных[^1_9][^1_8]
- Система вознаграждений на основе PnL и Sharpe Ratio[^1_11][^1_8]
- Адаптация к изменяющимся рыночным условиям[^1_12][^1_11]


## Детальная архитектура файловой структуры

### feature_engineering.py - Модернизированная инженерия признаков

**Текущая функциональность:**

- Извлечение технических индикаторов из OHLCV данных
- Создание паттерн-векторов для xLSTM

**Новая функциональность:**

```python
class AdvancedFeatureEngineer:
    def __init__(self):
        self.vsa_analyzer = VSAAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        
    def extract_vsa_features(self, ohlcv_data):
        """
        Извлечение VSA-признаков для анализа умных денег
        """
        # Базовые VSA компоненты
        volume = ohlcv_data['volume']
        spread = ohlcv_data['high'] - ohlcv_data['low']  # Спред
        closing_position = (ohlcv_data['close'] - ohlcv_data['low']) / spread
        
        # VSA сигналы
        no_demand_bars = self.detect_no_demand(volume, spread, closing_position)
        no_supply_bars = self.detect_no_supply(volume, spread, closing_position)
        stopping_volume = self.detect_stopping_volume(volume, spread)
        climactic_volume = self.detect_climactic_action(volume, spread)
        
        # Фазы накопления/распределения
        accumulation_phase = self.detect_accumulation(volume, spread, ohlcv_data)
        distribution_phase = self.detect_distribution(volume, spread, ohlcv_data)
        
        return {
            'vsa_signals': [no_demand_bars, no_supply_bars, stopping_volume, climactic_volume],
            'smart_money_flow': [accumulation_phase, distribution_phase],
            'volume_strength': self.calculate_volume_strength(volume, spread),
            'price_volume_relationship': closing_position
        }
    
    def create_xlstm_input_vectors(self, technical_indicators, vsa_features):
        """
        Создание входных векторов для xLSTM с объединением индикаторов и VSA
        """
        # Объединение традиционных индикаторов с VSA
        combined_features = np.concatenate([
            technical_indicators,
            vsa_features['vsa_signals'],
            vsa_features['smart_money_flow'],
            vsa_features['volume_strength']
        ])
        
        # Нормализация для xLSTM
        normalized_features = self.normalize_for_xlstm(combined_features)
        
        return normalized_features
```

**VSA-индикаторы для интеграции:**[^1_6][^1_2][^1_3][^1_7][^1_1]

- **No Demand Bar**: узкий спред вверх при низком объеме
- **No Supply Bar**: узкий спред вниз при низком объеме
- **Stopping Volume**: высокий объем при узком спреде после падения
- **Climactic Volume**: экстремальный объем на широком спреде
- **Testing**: низкообъемные бары после высокообъемных для тестирования силы


### train_model.py - Гибридная тренировка xLSTM + RL

**Текущая функциональность:**

- Тренировка двух отдельных xLSTM моделей
- Обучение на исторических данных

**Новая архитектура:**

```python
class HybridTrainingSystem:
    def __init__(self):
        self.xlstm_model = self.build_xlstm_architecture()
        self.rl_agent = self.build_rl_agent()
        self.vsa_processor = VSAProcessor()
        
    def build_xlstm_architecture(self):
        """
        Создание единой xLSTM модели с mLSTM и sLSTM компонентами
        """
        model = Sequential([
            # sLSTM слой с scalar memory и memory mixing
            sLSTMLayer(
                units=256,
                memory_cells=4,
                exponential_gating=True,
                memory_mixing=True
            ),
            
            # mLSTM слой с matrix memory для параллелизации  
            mLSTMLayer(
                units=128,
                matrix_memory_size=(32, 32),
                covariance_update_rule=True,
                parallel_training=True
            ),
            
            # Residual connections для xLSTM blocks
            ResidualBlock([
                sLSTMLayer(units=128),
                mLSTMLayer(units=64)
            ]),
            
            Dense(64, activation='tanh'),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # Buy, Sell, Hold
        ])
        
        return model
    
    def build_rl_agent(self):
        """
        Создание RL-агента для принятия торговых решений
        """
        # DQN агент с replay buffer
        state_size = 100  # размер входного состояния
        action_size = 3   # Buy, Sell, Hold
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=32
        )
        
        return agent
    
    def train_hybrid_system(self, training_data):
        """
        Совместное обучение xLSTM и RL компонентов
        """
        for epoch in range(self.epochs):
            # 1. Обучение xLSTM на VSA + технических индикаторах
            xlstm_predictions = self.train_xlstm_step(training_data)
            
            # 2. Использование предсказаний xLSTM как входа для RL
            rl_states = self.create_rl_states(xlstm_predictions, training_data)
            
            # 3. Обучение RL агента
            rewards = self.calculate_trading_rewards(training_data)
            self.rl_agent.train(rl_states, rewards)
            
            # 4. Обратная связь от RL к xLSTM (опционально)
            rl_feedback = self.rl_agent.get_action_values()
            self.adjust_xlstm_weights(rl_feedback)
```

**Ключевые улучшения xLSTM:**[^1_13][^1_4][^1_5]

- **Exponential Gating**: экспоненциальные функции активации вместо сигмоид
- **Matrix Memory (mLSTM)**: параллельные вычисления с ковариационным обновлением
- **Scalar Memory (sLSTM)**: улучшенное смешивание памяти между ячейками
- **Residual Blocks**: стекирование xLSTM блоков для глубокого обучения


### run_live_trading.py - Интеграция всех компонентов

**Новая архитектура:**

```python
class LiveTradingEngine:
    def __init__(self):
        self.xlstm_model = load_model('xlstm_vsa_model.h5')
        self.rl_agent = load_rl_agent('rl_agent.pkl')
        self.vsa_analyzer = VSAAnalyzer()
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # RL Environment для live trading
        self.trading_env = TradingEnvironment()
        
    def process_live_data(self, live_ohlcv):
        """
        Обработка живых данных через всю pipeline
        """
        # 1. Извлечение VSA признаков
        vsa_features = self.feature_engineer.extract_vsa_features(live_ohlcv)
        
        # 2. Создание входных векторов для xLSTM
        technical_indicators = self.extract_technical_indicators(live_ohlcv)
        xlstm_input = self.feature_engineer.create_xlstm_input_vectors(
            technical_indicators, vsa_features
        )
        
        # 3. Предсказание через xLSTM
        xlstm_prediction = self.xlstm_model.predict(xlstm_input)
        
        # 4. Создание состояния для RL агента
        rl_state = np.concatenate([
            xlstm_prediction.flatten(),
            vsa_features['vsa_signals'],
            [self.current_position, self.current_pnl]
        ])
        
        # 5. Принятие торгового решения через RL
        action = self.rl_agent.act(rl_state)
        
        return self.execute_trading_action(action, vsa_features)
    
    def execute_trading_action(self, action, vsa_features):
        """
        Выполнение торгового действия с учетом VSA контекста
        """
        # Проверка VSA условий перед торговлей
        if action == 'BUY':
            # Проверяем наличие накопления или stopping volume
            if vsa_features['smart_money_flow'][^1_0] > 0.7:  # accumulation
                return self.place_buy_order()
            
        elif action == 'SELL':
            # Проверяем наличие распределения или climactic volume
            if vsa_features['smart_money_flow'][^1_1] > 0.7:  # distribution
                return self.place_sell_order()
                
        return 'HOLD'
```


### trade_manager.py - Усовершенствованное управление позициями

**Новая функциональность:**

```python
class AdvancedTradeManager:
    def __init__(self):
        self.rl_position_sizer = RLPositionSizer()
        self.vsa_risk_manager = VSARiskManager()
        
    def calculate_position_size(self, signal_strength, vsa_confirmation, rl_confidence):
        """
        Динамический расчет размера позиции на основе всех сигналов
        """
        base_size = self.account_balance * 0.02  # 2% риск
        
        # Корректировка на основе VSA подтверждения
        vsa_multiplier = 1.0
        if vsa_confirmation['accumulation'] > 0.8:
            vsa_multiplier = 1.5  # Увеличиваем позицию при сильном накоплении
        elif vsa_confirmation['distribution'] > 0.8:
            vsa_multiplier = 0.5  # Уменьшаем при распределении
            
        # Корректировка на основе RL уверенности
        rl_multiplier = rl_confidence
        
        final_size = base_size * vsa_multiplier * rl_multiplier
        
        return min(final_size, self.max_position_size)
    
    def set_dynamic_stops(self, vsa_analysis, rl_volatility_prediction):
        """
        Динамические стоп-лоссы на основе VSA и RL анализа
        """
        if vsa_analysis['stopping_volume']:
            # При stopping volume можно использовать более тайтовые стопы
            stop_distance = self.current_price * 0.005  # 0.5%
        elif vsa_analysis['climactic_volume']:
            # При climactic volume нужны более широкие стопы
            stop_distance = self.current_price * 0.02   # 2%
        else:
            # Базовый расчет через RL предсказание волатильности
            stop_distance = self.current_price * rl_volatility_prediction
            
        return stop_distance
```


## Преимущества новой архитектуры

### Синергия компонентов

**1. xLSTM + VSA интеграция:**[^1_13][^1_1][^1_5]

- xLSTM обрабатывает последовательности VSA-паттернов во времени
- Матричная память mLSTM позволяет параллельно анализировать множественные VSA сигналы
- Экспоненциальные ворота лучше фильтруют VSA-шум от значимых сигналов

**2. VSA + RL синергия:**[^1_14][^1_15][^1_8]

- VSA предоставляет качественный контекст для RL агента
- RL агент адаптируется к различным VSA режимам рынка
- Система вознаграждений учитывает подтверждение VSA сигналов

**3. xLSTM + RL интеграция:**[^1_16][^1_8][^1_9]

- xLSTM предсказания становятся частью состояния RL агента
- RL агент принимает окончательные торговые решения
- Непрерывное обучение RL компонента на живых данных


### Улучшения по сравнению с текущим ботом

**Вместо двух отдельных xLSTM:**

- Единая мощная модель с расширенной архитектурой
- Матричная память для лучшего моделирования зависимостей
- Экспоненциальные ворота для улучшенного контроля информации

**Добавление VSA анализа:**[^1_2][^1_3][^1_1]

- Понимание намерений институциональных игроков
- Анализ фаз накопления и распределения
- Контекстная информация о силе и слабости рынка
- Опережающие сигналы вместо запаздывающих индикаторов

**Интеграция RL:**[^1_8][^1_11][^1_9]

- Адаптация к изменяющимся рыночным условиям
- Непрерывное обучение на новых данных
- Оптимизация риск-доходность в реальном времени
- Способность к комплексному принятию решений


## Технические рекомендации для реализации

### Этапы разработки

**Этап 1: VSA-модуль**

1. Реализовать базовые VSA индикаторы[^1_1][^1_2]
2. Создать детекторы накопления/распределения[^1_3][^1_7]
3. Интегрировать с существующими техническими индикаторами
4. Провести backtesting VSA сигналов

**Этап 2: Модернизация xLSTM**

1. Внедрить sLSTM компонент с memory mixing[^1_5]
2. Добавить mLSTM с matrix memory[^1_5]
3. Реализовать exponential gating[^1_5]
4. Создать residual blocks архитектуру[^1_5]

**Этап 3: RL интеграция**

1. Выбрать RL алгоритм (DQN, PPO, A2C)[^1_9][^1_8]
2. Определить пространство состояний и действий[^1_11]
3. Создать систему вознаграждений[^1_8][^1_11]
4. Реализовать continuous learning pipeline[^1_9]

**Этап 4: Интеграция и тестирование**

1. Объединить все компоненты в единую систему
2. Провести комплексное backtesting
3. Paper trading для валидации
4. Постепенный переход к live trading

### Рекомендуемые библиотеки

**xLSTM реализация:**[^1_17][^1_18][^1_19]

```python
# PyxLSTM - готовая реализация xLSTM
pip install PyxLSTM

# Альтернативно - официальная реализация NX-AI
git clone https://github.com/NX-AI/xlstm.git
pip install xlstm
```

**RL библиотеки:**[^1_20][^1_8]

```python
# Stable-baselines3 для RL алгоритмов
pip install stable-baselines3

# Gym для создания trading environment  
pip install gymnasium

# Ray RLLib для масштабируемого RL
pip install ray[rllib]
```

**VSA и технический анализ:**[^1_21][^1_22]

```python
# TA-Lib для технических индикаторов
pip install TA-Lib

# Custom VSA реализация (нужно создать самостоятельно)
```

Данная архитектура представляет существенное улучшение вашей текущей системы, объединяя предсказательную силу xLSTM, понимание рыночной микроструктуры через VSA и адаптивность RL в единую интеллектуальную торговую систему.
<span style="display:none">[^1_100][^1_101][^1_102][^1_103][^1_104][^1_105][^1_106][^1_107][^1_108][^1_109][^1_110][^1_111][^1_112][^1_113][^1_114][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div style="text-align: center">⁂</div>

[^1_1]: https://blog.opofinance.com/en/volume-spread-analysis/

[^1_2]: https://justmarkets.com/trading-articles/learning/what-is-the-vsa-trading-strategy

[^1_3]: https://chartswatcher.com/pages/blog/volume-spread-analysis-tutorial-boost-trading

[^1_4]: https://arxiv.org/pdf/2405.04517.pdf

[^1_5]: https://proceedings.neurips.cc/paper_files/paper/2024/file/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Paper-Conference.pdf

[^1_6]: https://fundyourfx.com/using-volume-spread-analysis-to-identify-market-strength-and-weakness/

[^1_7]: https://www.stockgro.club/blogs/trading/volume-spread-analysis/

[^1_8]: https://blog.quantinsti.com/reinforcement-learning-trading/

[^1_9]: https://arxiv.org/html/2411.07585v1

[^1_10]: https://www.reddit.com/r/algotrading/comments/viag8p/reinforcement_learning_for_algorithmic_trading/

[^1_11]: https://milvus.io/ai-quick-reference/how-does-reinforcement-learning-work-in-financial-trading

[^1_12]: https://tickeron.com/blogs/ai-trading-in-2025-how-bots-and-machine-learning-transform-stock-markets-11468/

[^1_13]: https://arxiv.org/html/2505.05325v1

[^1_14]: https://blog.mlq.ai/deep-reinforcement-learning-trading-strategies-automl/

[^1_15]: https://www.timothysykes.com/blog/volume-spread-analysis/

[^1_16]: https://python.plainenglish.io/python-powered-algorithmic-trading-in-2025-reinforcement-learning-ai-and-beyond-0c5db4fb31d5

[^1_17]: https://github.com/muditbhargava66/PyxLSTM

[^1_18]: https://github.com/gonzalopezgil/xlstm-ts

[^1_19]: https://github.com/NX-AI/xlstm

[^1_20]: https://github.com/stefan-jansen/machine-learning-for-trading

[^1_21]: https://www.luxalgo.com/blog/feature-engineering-in-trading-turning-data-into-insights/

[^1_22]: https://github.com/asavinov/intelligent-trading-bot

[^1_23]: https://www.xchief.com/id/library/indicators/volume-spread-analysis-indicator/

[^1_24]: https://www.forex-ratings.com/technical-analysis/decoding-volume-exploring-volume-spread-analysis-vsa-in-forex-trading/

[^1_25]: https://tradersunion.com/technic-analysis/volume-spread-analysis/

[^1_26]: https://github.com/TopTalent-23/Volume-Spread-Analysis-VSA-indicator-for-Metatrader-5

[^1_27]: https://www.writofinance.com/volume-spread-analysis-vsa/

[^1_28]: https://clusterdelta.com/ru/baza-znanij/voltheory/section-9

[^1_29]: https://ru.tradingview.com/scripts/vsa/

[^1_30]: https://www.ebc.com/ru/webinars/course-replays/20250507_001

[^1_31]: https://realtrading.com/trading-blog/volume-spread-analysis-5-key-concepts/

[^1_32]: https://ru.tradingview.com/scripts/vsa/page-3/?script_access=all\&sort=recent

[^1_33]: https://www.tradeguider.com/resource_center1.asp

[^1_34]: https://www.instaforex.com/knowledge_base/466-volume-spread-analysis

[^1_35]: https://fxcodebase.com/code/viewtopic.php?f=17\&t=62307\&start=20

[^1_36]: https://www.youtube.com/watch?v=jNh8j8uFmJs

[^1_37]: https://ftmo.com/en/volume-spread-analysis/

[^1_38]: https://www.atlantis-press.com/article/125999560.pdf

[^1_39]: https://coredevsltd.com/articles/5-best-ai-trading-bots-in-2025/

[^1_40]: https://arxiv.org/html/2411.12746v1

[^1_41]: https://www.tokenmetrics.com/blog/top-crypto-trading-bots-2025-open-source-paid-compared?74e29fd5_page=18

[^1_42]: https://neptune.ai/blog/7-applications-of-reinforcement-learning-in-finance-and-trading

[^1_43]: https://www.malgotechnologies.com/top-10-crypto-trading-bots-2025

[^1_44]: https://www.reinforcementlearningpath.com/choosing-rl-algorithm/

[^1_45]: https://www.econstor.eu/handle/10419/183139

[^1_46]: https://www.coinapi.io/blog/best-ai-crypto-trading-bots-for-2025

[^1_47]: https://www.sciencedirect.com/science/article/pii/S2590005625000177

[^1_48]: https://www.sciencedirect.com/science/article/abs/pii/S0957417423023515

[^1_49]: https://www.youtube.com/watch?v=hDhFRT8DcGY

[^1_50]: https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/

[^1_51]: https://ijrpr.com/uploads/V6ISSUE5/IJRPR47353.pdf

[^1_52]: https://library.fiveable.me/deep-learning-systems/unit-9/lstm-architecture-gating-mechanisms/study-guide/BKlYA4fD924yTeP6

[^1_53]: https://www.ijsat.org/papers/2025/2/3868.pdf

[^1_54]: https://wire.insiderfinance.io/long-short-term-memory-lstm-networks-for-predicting-market-trends-in-high-frequency-trading-d65fa40e21b9

[^1_55]: https://d2l.ai/chapter_recurrent-modern/lstm.html

[^1_56]: https://science.lpnu.ua/sites/default/files/journal-paper/2025/feb/38149/2025124158167.pdf

[^1_57]: https://github.com/jiewwantan/RNN_LSTM_trading_model

[^1_58]: https://milvus.io/ai-quick-reference/what-are-long-shortterm-memory-lstm-networks

[^1_59]: https://www.sciencedirect.com/science/article/abs/pii/S1568494625005526

[^1_60]: https://www.atlantis-press.com/article/126011813.pdf

[^1_61]: https://apxml.com/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm/lstm-updating-cell-state

[^1_62]: https://journals.mriindia.com/index.php/ijacte/article/view/206

[^1_63]: https://www.linkedin.com/pulse/long-short-term-memory-lstm-bluechip-technologies-asia-udacc

[^1_64]: https://en.wikipedia.org/wiki/Long_short-term_memory

[^1_65]: https://faba.bg/index.php/faba/article/view/265/137

[^1_66]: https://apxml.com/courses/foundations-transformers-architecture/chapter-1-revisiting-sequence-modeling-limitations/lstm-gating-mechanisms

[^1_67]: https://arxiv.org/abs/2503.13817

[^1_68]: https://diposit.ub.edu/dspace/bitstream/2445/182488/2/tfg_johnny_nu%C3%B1ez_cano.pdf

[^1_69]: https://arxiv.org/html/2411.06389v1

[^1_70]: https://arxiv.org/abs/2302.12689

[^1_71]: https://www.sciencedirect.com/science/article/pii/S0957417424013319

[^1_72]: https://www.sciencedirect.com/science/article/abs/pii/S136184152300083X

[^1_73]: https://www.nature.com/articles/s41598-025-12516-3

[^1_74]: https://www.youtube.com/watch?v=_wZQR3vTzwo

[^1_75]: https://github.com/cashiu/awesome-deep-reinforcement-learning-in-finance

[^1_76]: https://www.scribd.com/document/839917829/guides-volume-spread-analysis-Support-and-resistance

[^1_77]: https://www.sciencedirect.com/science/article/pii/S0950705124011559

[^1_78]: https://skyforbes.com/how-to-use-volume-spread-analysis-vsa-for-forex-trading/

[^1_79]: https://www.studocu.com/row/document/american-international-university-bangladesh/algorithms/rise-precision-volume-spread-analysis/99337629

[^1_80]: https://atas.net/volume-analysis/cluster-analysis-and-vsa/

[^1_81]: https://www.geeksforgeeks.org/deep-learning/tf-keras-layers-lstm-in-tensorflow/

[^1_82]: https://wire.insiderfinance.io/the-algorithmic-trading-secret-lstm-neural-networks-on-metatrader-5-revealed-ab2f2d5b19cb

[^1_83]: https://ml-tutorials.readthedocs.io/en/latest/auto_examples/lstm_tensorflow_scratch.html

[^1_84]: https://github.com/zach1502/LSTM-Algorithmic-Trading-Bot

[^1_85]: https://github.com/dtunai/xLSTM-Jax

[^1_86]: https://keras.io/api/layers/recurrent_layers/lstm/

[^1_87]: https://www.youtube.com/watch?v=DM7xyNCGyB0

[^1_88]: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM

[^1_89]: https://www.kaggle.com/code/fedewole/algorithmic-trading-with-keras-using-lstm

[^1_90]: https://www.nx-ai.com/en/news/xlstm-source-code-now-open-source

[^1_91]: https://medium.datadriveninvestor.com/implementing-lstm-networks-with-python-a-guide-using-tensorflow-and-keras-915b58f502ce

[^1_92]: https://codesandbox.io/s/allenye66-lstm-stock-trading-bot-zkcz96

[^1_93]: https://github.com/nikolaikyhne/xlstm-senet

[^1_94]: https://stackoverflow.com/questions/67304916/lstm-with-high-sequence-length-tensorflow-and-keras

[^1_95]: https://ai.plainenglish.io/how-i-trained-an-ai-trading-bot-to-outsmart-my-own-strategies-c16f3cd9783e

[^1_96]: https://github.com/robot-bulls/Tensorflow_XLSTM

[^1_97]: https://python-forum.io/thread-44197.html

[^1_98]: https://www.techscience.com/cmc/v83n2/60595/html

[^1_99]: https://www.rapidinnovation.io/post/pattern-recognition-in-ml-a-comprehensive-overview

[^1_100]: https://arxiv.org/abs/2310.09903

[^1_101]: https://pareto.ai/blog/pattern-recognition-in-machine-learning

[^1_102]: https://syntiumalgo.com/feature-engineering-for-ai-trading/

[^1_103]: https://arxiv.org/pdf/2310.09903.pdf

[^1_104]: https://www.v7labs.com/blog/pattern-recognition-guide

[^1_105]: https://actingintelligent.com/practical-feature-selection-for-algorithmic-trading/

[^1_106]: https://www.sciencedirect.com/science/article/pii/S2666827025000143

[^1_107]: https://serokell.io/blog/applications-of-pattern-recognition

[^1_108]: https://www.youtube.com/watch?v=FUB1KlhqH58

[^1_109]: https://www.scitepress.org/Papers/2025/134813/134813.pdf

[^1_110]: https://www.sciencedirect.com/science/article/pii/S0957417424003865

[^1_111]: https://coinsbench.com/what-you-need-to-build-an-automated-ai-crypto-trading-bot-56a82b0c60cb

[^1_112]: https://consensus.app/search/feature-selection-methods-for-technical-indicators/axPgmeYETdulw8L2dAHFew/

[^1_113]: https://www.mql5.com/en/articles/16230

[^1_114]: https://www.sciencedirect.com/science/article/pii/S0957417425010036

