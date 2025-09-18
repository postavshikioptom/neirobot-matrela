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