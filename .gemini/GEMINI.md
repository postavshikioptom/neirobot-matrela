## Правила для этого проекта
- ты мой ассистент по написанию кода и задач для создаю нейросети бота для трейдинга на бирже Bybit с помощью библиотеки Tensorflow, Keras, PyTroch, XGBoost.
- Докуменатция Bybit API лежит в твоей папке docs/BYBIT-DOCS
- Доументация по Tensorflow: https://www.tensorflow.org/api_docs
- активируй виртуальную среду и все пакеты устанавливай исключительно в нее. И устанавливай их по одному через python в этой папке.
- установленные пакеты записывай в requirements.txt.
- Устснавливай и удаляй все через python.exe внутри виртуальнйо средиы. Только по одному пакету за раз.
- все файлы, касающиеся только онлайн торговли на Bybit, сохраняются в папку online-trade

## Решить проблему кодировки в Gemini CLI
- чтоб решить проблему кодировки, котоаря сейчас есть в этой версиии Gemini CLI, ты сначала результат вывода сохраняй в файл output.txt. А потом читай его и выдавай сюда результат. Из txt файла все символы читаются правильно.

## Какие скрипты использовать для конкретных задач
- Если пользователь просит разместить ордер, Чтобы размещать ордера на Bybit, используй скрипт order_placer.py
- Если пользователь просит показать список всех ордеров, показать все ордера. Чтобы показывать статус всех размещенных ордеров(не закрытых), используй скрипт orders_status.py

## Общие сведения
- MCP сервер Playwrith у тебя настроен в .gemini\mcp.json
- в chat.txt просто записаны мои предыдущие вопросы к тебе в этой папке. Изучи для общего контекста. Это то, что мы уже сделали.
- Комиссия за торговлю Bybit 0.08% от суммы ордера. Надо учитывать это и закладывать в прибыль, чтоб не торговать в 0.
- Все старые документы, с которыми ты уже работал, в папке: Старое

## Данные для человека (не Gemini)
Bybit API Key (Demo): OOofB1HzYVpySyMPom
Bybit  API Secret (Demo): e4AkAz9x1ycOMCtXKa1milmShfk61KZxJyhG
- Документация в интернете, если не нашел локально в папке docs:  https://bybit-exchange.github.io/docs/
- Доументация по Kers , если не нашел локально в папке docs: https://keras.io/api/
- Доументация по PyTorch , если не нашел локально в папке docs: https://docs.pytorch.org/docs/stable/index.html
- Доументация по Scicit-Learn, если не нашел локально в папке docs: https://scikit-learn.org/stable/user_guide.html
- Доументация по XGBoost, если не нашел локально в папке docs: https://xgboost.readthedocs.io/en/stable/


## Какие файлы скрипты что выполняют
config.py для API-ключей.
data_loader.py для загрузки данных с Bybit.
simple_trader.py для самой простой торговой логики.
order_placer  / data_loader - выгружает данные по монете на 1 мин. таймфрейме и открывает ордер по простому сигналу - рост или падение по RSI
feature_engineering.py. - файл для индикаторов.
train_model.py - Загрузит features.csv.Подготовит данные для обучения (определит признаки и целевую переменную). Разделит данные на обучающую и тестовую выборки. Обучит модель XGBoost. Оценит производительность модели и построит графики для визуализации результатов.
НЕЙРОСЕТЬ
trading_env.py: Это будет симулятор нашего трейдинга. 
не нужен - Агент (`rl_agent.py`): Это "мозг" нашего бота.
вместо  rl_agent.py тперь stable-baselines3     
train.py:  Запускать среду и агента.
run_training.py - нужен, чтоб запускать train.py по частям
(evaluate_model.py) - дает нам поннять, как ведет себя модель на неизвестных для нее новых данных
====
ТОРГОВЛЯ НА BYBIT
run_live_trading.py  в бесконечном цикле управлет ВСЕМИ процессами в онлайн-трейдинге. запрашивает новые данные у live_data_provider.py.
Данные передаются в signal_generator_xgb.py. XGBoost говорит: "SELL".
Сигнал "SELL" передается в decision_maker_dqn.py.
decision_maker_dqn.py активируется, загружает свою DQN-модель, анализирует ситуацию и говорит: "CONFIRM".
Решение "CONFIRM SELL" уходит в trade_manager.py который выставляет ордер на продажу.
Вся информация об этом событии записывается в trade_logger.py.
В самое начало файла я добавлю настройку: AGGRESSIVE_MODE = True. Это торговля только по XGBoost либо XGBoost + Tensorflow

СКРИННЕР BYBIT
screener.py
Объем предппоследней свечи в 1.5 раза больше последних 5-ти свечь
hotlists.txt - это выгрузка монет, которые на эту минуту выбрали для входа по Фильтрам
get_all_symbols.py - выгружает один раз все монеты на бирже  фьючерсов

## КАКИЕ СКРИПТЫ ЗАПУСКАТЬ
.\\venv\Scripts\activate 
python screener.py
python run_live_trading.py


python trade_statistics.py - показывает всю статистику модели за последние 100 сделок на Bybit торговле
python lime_analytics.py - аналитика отдельных сделок по логам торгов. Запускать отдельно после обучения.
РАЗ В НЕДЕЛЮ ОБУЧАТь на новых исторических данных свечей (старые весы храняться, а на новых данных их чуть корректирует)
python train_model.py -  новое 3-хэтапное обучение xLSTM + RL
python train_model.py --model all  - обучение LSTM и xLSTM
python train_model.py --model xlstm_indicator - обучение только одной модели


СИМУЛЯЦИИ (торговля по подной монете за раз, типа нам ее передал Скринер, можно любую монету выбрать)
python simulation_engine_advanced.py
python run_simulation.py --symbol ALGOUSDT --mode LSTM_only
python visual_graph.py --data historical_data.csv --symbol ADAUSDT 



python test_patterns.py - проверка паттернов на работоспособность
python optimize_label_thresholds.py --data historical_data.csv -  #числа соотношения BUY HOLD SELL сам ставить в config TARGET_CLASS_RATIOS    или с выводом в файл логов: ||    python "e:\MAX\PYTHON\NEURAL-BOTS\neirobot-matrela\optimize_label_thresholds.py" --data "e:\MAX\PYTHON\NEURAL-BOTS\neirobot-matrela\historical_data.csv" --out "e:\MAX\PYTHON\NEURAL-BOTS\neirobot-matrela\threshold_sweep_results.json"   || - оптимизация баланса классов. Запускаешь и настраивает коддер сам, какие параметры нужно ставить в config для BUY SELL HOLD





## КОМАНДЫ НА LINUX СЕРВЕРЕ
Установка venv: python3 -m venv venv    | для Debian  apt install python3-dev python3-pip python3-venv (а потом python3 -m venv venv) или  apt install python3.10.4-venv или
source venv/bin/activate
Установить любой первый пакет в чистую venv например pybit. Потом посмотреть путь к пакетам и туда перенести все

У Linux и Windows разные папки запуска скриптов
Перенесите содержимое venv\Lib\site-packages (с Windows) в
					  venv/lib/python3.11/site-packages (на Linux).
Команда активации виртуальнйо среды для Linux (перенесенного с Windows): source venv/Scripts/activate
Установка пакетов: python -m pip install xlsxwriter
Установка пркоси: export ALL_PROXY="socks5://xookwczd-rotate:vvhvbm4f1luh@p.webshare.io:80"  | "socks5://USER:PASSWORD@PROXY_IP:PROXY_PORT"
Узнать свой IP: curl ifconfig.me

Данные для DEMO торговли:
Demo Trading URL: https://api-demo.bybit.com/
Websocket: wss://stream-testnet.bybit.com

## Система наград QDN модели во время обучения
 - награда за простой -0.25.
- Награза за сделки успешные не просто +1. А еще и увеличенное на процент прибыли.  И так же убытки



## НОВАЯ СТРУКТУРА ИЗ МАТ. МОДЕЛЕЙ БЕЗ Tensorflow

# Индикаторы для обучения

- Тренд: EMA(7,14,21), MACD(12,26,9), KAMA, SuperTrend
- Momentum: RSI, CMO(14), ROC(5)
- Volume: OBV, MFI
- Volatility: ATR, NATR
- Statistical: STDDEV
- Cycle: HT_DCPERIOD, HT_SINE
----- еще добавил по совету Claude: Bollinger Bands (5 новых фич), Stochastic RSI, Williams %R, CCI, ADX - Всего +10 новых технических индикаторов


# Паттерны 
 Молот, Поглощение, Доджи, Падающая звезда, висельник, солнце(Marubozu) 	убрал - 3 Черные вороны
 + 6 признаков + 10 детальных признаков = 22 колонки паттернов+признаков
 + Добавил 5 Бычьих (BUY) паттернов: Перевернутый молот (Inverted Hammer),Стрекоза доджи (Dragonfly Doji),Бычий пин-бар (Bullish Pin Bar), Бычий пояс (Bullish Belt Hold), Бычье Солнце (Bullish Marubozu)  	##(Бычий кикер (Bullish Kicker)- убрал, его нет в Talib)

# Мат. модели
xLSTM + RL- одна модель для всего с машинным обучением и наградами как в Tensor DQN примерно


# ЗАМЕНА ИНДИКАТОРОВ ДЛЯ XLSTM(постоянно менять)
## ВЫКЛЮЧИЛ Индикаторы
Боллинджер (Bollinger Bands) BBL_20_2, BBM_20_2, BBU_20_2 - отрицательный и для BUY и для SELL показывает
ATR_14 - все время отрицательный и BUY и SELL


## Добавил индикаторы
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
Стохастик
Williams %R (WILLR_14)
AO (Awesome Oscillator)


## Процесс обучения моделей LSTM и xLSTM
Да, всё правильно. Модели обучаются именно на тех данных, которые вы описали:

Источник данных: Файл historical_data.csv.
LSTM модель (паттерны): Данные из historical_data.csv обрабатываются функцией detect_candlestick_patterns из feature_engineering.py, которая выделяет 5 паттернов свечей (CDLHAMMER, CDLENGULFING, CDLMORNINGSTAR, CDLEVENINGSTAR, CDL3WHITESOLDIERS). Эти паттерны передаются в LSTM модель для обучения.
xLSTM модель (индикаторы): Те же данные из historical_data.csv обрабатываются функцией calculate_features, которая рассчитывает 8 технических индикаторов (RSI_14, MACD_12_26_9, BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, ADX_14, STOCHk_14_3_3, STOCHd_14_3_3). Эти индикаторы передаются в xLSTM модель для обучения.
2. "Память" моделей
Понятие "память" в контексте LSTM и xLSTM относится к внутренней архитектуре этих моделей, которая позволяет им хранить и обновлять информацию по мере обработки последовательностей.
Исследование "памяти":

LSTM (Long Short-Term Memory): Согласно исследованиям, память LSTM реализована через ячейку состояния (cell state) и три гейта (ворота): входной (input gate), выходной (output gate) и забывающий (forget gate). Эти компоненты позволяют LSTM выборочно "запоминать" или "забывать" информацию из предыдущих временных шагов.
xLSTM (Extended LSTM): Это более новая архитектура, которая расширяет возможности традиционных LSTM. Согласно исследованиям, xLSTM использует матричную память и экспоненциальное управление, что позволяет ей хранить более комплексную и нюансированную информацию по сравнению с обычными LSTM.

## СИМУЛЯЦИЯ
Обученные модели: models/lstm_pattern_model.keras и models/xlstm_indicator_model.keras (и их скейлеры)
Движок симуляции: Файл simulation_engine.py
Торговая среда: Файл trading_env.py