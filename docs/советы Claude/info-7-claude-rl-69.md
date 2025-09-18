Инструкция по настройке переключения между GPU и CPU
Изменения в файлах кода
1. Файл train_model.py (основной скрипт)
В начало файла добавьте следующий код:
import os
import tensorflow as tf

# Настройка переменных окружения для TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Убираем лишние предупреждения
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Динамическое выделение памяти

# Функция для настройки устройств
def setup_devices():
    try:
        # Проверяем доступность GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print(f"Найдено {len(physical_devices)} GPU устройств")
            # Настраиваем динамическое выделение памяти для всех GPU
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Включено динамическое выделение памяти для {device}")
            return True
        else:
            print("GPU не обнаружены, используется CPU")
            return False
    except Exception as e:
        print(f"Ошибка при настройке устройств: {e}")
        print("Переключение на CPU")
        return False

# Вызываем функцию настройки в начале программы
has_gpu = setup_devices()

Затем вызовите эту функцию в блоке main() перед инициализацией моделей.
2. Файл models/xlstm_rl_model.py
Измените метод __init__:
def __init__(self, input_shape, memory_size=64, memory_units=128, weight_decay=1e-4, gradient_clip_norm=1.0):
    # Существующий код...
    self.input_shape = input_shape
    self.memory_size = memory_size
    self.memory_units = memory_units
    self.weight_decay = weight_decay
    self.gradient_clip_norm = gradient_clip_norm
    
    # Настройка оптимизаторов с учетом доступного устройства
    self._configure_optimizers()
    
    # Остальной код...

def _configure_optimizers(self):
    """Настраивает оптимизаторы с учетом доступного устройства"""
    try:
        # Проверяем, доступны ли GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            # Для GPU используем более агрессивные настройки
            self.supervised_optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=self.gradient_clip_norm
            )
            self.actor_optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0005,
                clipnorm=self.gradient_clip_norm
            )
            self.critic_optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=self.gradient_clip_norm
            )
            print("Настроены оптимизаторы для GPU")
        else:
            # Для CPU используем более консервативные настройки
            self.supervised_optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0005,
                clipnorm=self.gradient_clip_norm
            )
            self.actor_optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0001,
                clipnorm=self.gradient_clip_norm
            )
            self.critic_optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0005,
                clipnorm=self.gradient_clip_norm
            )
            print("Настроены оптимизаторы для CPU")
    except Exception as e:
        print(f"Ошибка при настройке оптимизаторов: {e}")
        # Fallback на стандартные настройки
        self.supervised_optimizer = tf.keras.optimizers.Adam(clipnorm=self.gradient_clip_norm)
        self.actor_optimizer = tf.keras.optimizers.Adam(clipnorm=self.gradient_clip_norm)
        self.critic_optimizer = tf.keras.optimizers.Adam(clipnorm=self.gradient_clip_norm)

3. Файл ThreeStageTrainer.py (или где у вас класс ThreeStageTrainer)
Добавьте метод для определения размера батча в зависимости от устройства:
def _get_optimal_batch_size(self):
    """Определяет оптимальный размер батча в зависимости от устройства"""
    try:
        if tf.config.list_physical_devices('GPU'):
            # Для GPU можем использовать больший размер батча
            return config.SUPERVISED_BATCH_SIZE
        else:
            # Для CPU уменьшаем размер батча
            return max(16, config.SUPERVISED_BATCH_SIZE // 4)
    except:
        # В случае ошибки используем консервативный размер
        return 16

И используйте этот метод в stage1_supervised_pretraining:
def stage1_supervised_pretraining(self):
    # Существующий код...
    
    # Определяем оптимальный размер батча
    batch_size = self._get_optimal_batch_size()
    print(f"Используем размер батча: {batch_size}")
    
    history = self.model.actor_model.fit(
        self.X_train_supervised, self.y_train_supervised,
        validation_data=(self.X_val_supervised, self.y_val_supervised),
        epochs=config.SUPERVISED_EPOCHS,
        batch_size=batch_size,  # Используем динамический размер батча
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
    )
    
    # Остальной код...

4. Файл feature_engineering.py
Добавьте проверку на наличие GPU для оптимизации обработки данных:
def _calculate_all_indicators_batch(self, df):
    """Массовый расчет всех индикаторов с учетом доступного устройства"""
    try:
        # Проверяем доступность GPU для оптимизации
        has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        
        close_prices = df['close'].ffill().values
        high_prices = df['high'].ffill().values
        low_prices = df['low'].ffill().values
        
        # Если нет GPU, используем более эффективный подход с меньшим потреблением памяти
        if not has_gpu:
            # Вычисляем индикаторы по одному, очищая память после каждого
            indicators = {}
            
            # RSI
            indicators['RSI'] = talib.RSI(close_prices, timeperiod=config.RSI_PERIOD)
            gc.collect()  # Явно вызываем сборщик мусора после каждого тяжелого вычисления
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(
                close_prices,
                fastperiod=config.MACD_FASTPERIOD,
                slowperiod=config.MACD_SLOWPERIOD,
                signalperiod=config.MACD_SIGNALPERIOD
            )
            indicators['MACD'] = macd
            indicators['MACDSIGNAL'] = macdsignal
            indicators['MACDHIST'] = macdhist
            gc.collect()
            
            # И так далее для остальных индикаторов...
        else:
            # На GPU можем вычислять все сразу
            # Существующий код для вычисления индикаторов...
        
        # Добавляем все индикаторы в DataFrame
        for name, values in indicators.items():
            df[name] = values
        
        return True
            
    except Exception as e:
        print(f"Ошибка при массовом расчете индикаторов: {e}")
        return False

5. Создайте новый файл device_config.py для централизованного управления
# device_config.py
import os
import tensorflow as tf
import platform

class DeviceConfig:
    @staticmethod
    def setup():
        """Настраивает оптимальные параметры для TensorFlow в зависимости от среды"""
        # Определяем тип среды
        is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        is_aws = 'AWS_EXECUTION_ENV' in os.environ
        
        # Базовые настройки
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # Специфичные настройки для разных сред
        if is_kaggle:
            print("Обнаружена среда Kaggle")
            # Kaggle обычно имеет одну GPU, используем её полностью
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        elif is_aws:
            print("Обнаружена среда AWS")
            # AWS может иметь несколько GPU, используем все
            pass
        else:
            print(f"Обнаружена локальная среда: {platform.system()}")
            # Для локальной среды можно добавить специфичные настройки
        
        # Настройка устройств TensorFlow
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                print(f"Найдено {len(physical_devices)} GPU устройств:")
                for i, device in enumerate(physical_devices):
                    print(f"  GPU {i}: {device}")
                    tf.config.experimental.set_memory_growth(device, True)
                return True, len(physical_devices)
            else:
                print("GPU не обнаружены, используется CPU")
                return False, 0
        except Exception as e:
            print(f"Ошибка при настройке устройств: {e}")
            print("Переключение на CPU")
            return False, 0
    
    @staticmethod
    def get_optimal_batch_size(base_size=32):
        """Возвращает оптимальный размер батча в зависимости от устройства"""
        try:
            gpus = len(tf.config.list_physical_devices('GPU'))
            if gpus > 0:
                # Для GPU увеличиваем размер батча пропорционально количеству GPU
                return base_size * max(1, gpus)
            else:
                # Для CPU уменьшаем размер батча
                return max(16, base_size // 2)
        except:
            return base_size // 2

6. Использование device_config.py в main.py или основном скрипте
from device_config import DeviceConfig

# В начале программы
has_gpu, num_gpus = DeviceConfig.setup()

# Передаем информацию о устройствах в ThreeStageTrainer
trainer = ThreeStageTrainer(data_path, has_gpu=has_gpu, num_gpus=num_gpus)

Дополнительные рекомендации


Для запуска на Kaggle: Добавьте в начало ноутбука:
# Проверка GPU на Kaggle
!nvidia-smi
from device_config import DeviceConfig
has_gpu, num_gpus = DeviceConfig.setup()



Для запуска на AWS: Убедитесь, что используете правильный AMI с поддержкой CUDA и добавьте:
# Проверка настройки AWS
!pip install tensorflow==2.10.0  # Версия, совместимая с CUDA на вашем AMI
from device_config import DeviceConfig
has_gpu, num_gpus = DeviceConfig.setup()



Для локальной среды без GPU: Код будет автоматически использовать CPU с оптимизированными параметрами для снижения нагрузки на память.


Эти изменения позволят вашему коду эффективно работать как на GPU, так и на CPU, автоматически определяя доступные ресурсы и оптимизируя параметры под конкретную среду выполнения.