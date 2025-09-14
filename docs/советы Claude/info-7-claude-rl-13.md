Хорошо, я вижу ошибку: TypeError: XLSTMRLModel.train() got an unexpected keyword argument 'class_weight'.
Это означает, что мы добавили аргумент class_weight в вызов xlstm_model.train() в файле train_model.py, но сам метод train в классе XLSTMRLModel (файл xlstm_rl_model.py) не ожидает этот аргумент.
Мы хотели передать class_weight непосредственно в model.fit() внутри XLSTMRLModel.train().
Вот как это исправить:
🔧 Инструкции по изменению (файл: xlstm_rl_model.py)


Измените сигнатуру метода train:

Откройте xlstm_rl_model.py.
Найдите метод train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, custom_callbacks=None).
Добавьте class_weight=None в сигнатуру этого метода.

# В xlstm_rl_model.py, в классе XLSTMRLModel:
# ...
def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, custom_callbacks=None, class_weight=None): # <--- ДОБАВЛЕНО: class_weight=None
    """Обучение с улучшенной стабильностью"""
    if self.model is None:
        self.build_model()
    
    # ... (существующий код) ...



Передайте class_weight в self.model.fit():

Внутри того же метода train, найдите вызов self.model.fit(...).
Добавьте class_weight=class_weight в аргументы этого вызова.

# В xlstm_rl_model.py, в классе XLSTMRLModel, в методе train(...):
# ...
    # Обучение с нормализованными данными
    history = self.model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,  # <--- ИЗМЕНЕНО: передаем class_weight
        callbacks=callbacks,
        verbose=0,
        shuffle=True
    )
    
    self.is_trained = True
    return history
# ...



🔧 Инструкции по изменению (файл: train_model.py)


Удалите локальное вычисление class_weight_dict:

Откройте train_model.py.
Найдите функцию train_xlstm_rl_system.
Удалите весь блок кода, который вычисляет y_integers, class_weights_array, class_weight_dict и дополнительное усиление весов. Этот код теперь должен быть в xlstm_rl_model.py.

# В train_model.py, в функции train_xlstm_rl_system(...):
# ...
# УДАЛИТЕ ЭТОТ БЛОК КОДА:
# ДОБАВЬТЕ: Вычисление весов классов для борьбы с дисбалансом
# from sklearn.utils.class_weight import compute_class_weight
# y_integers = np.argmax(y_train, axis=1) # Преобразуем one-hot в целые числа
# class_weights_array = compute_class_weight(
#     'balanced',
#     classes=np.unique(y_integers),
#     y=y_integers
# )
# class_weight_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}

# # НОВЫЙ БЛОК: ДОПОЛНИТЕЛЬНОЕ УСИЛЕНИЕ ВЗВЕШИВАНИЯ КЛАССОВ BUY/SELL
# # Умножим веса BUY и SELL на дополнительный коэффициент
# # Это заставит модель еще больше "страдать" от ошибок на этих классах
# additional_weight_multiplier = 1.5 # Можно экспериментировать: 1.2, 1.5, 2.0
# if 0 in class_weight_dict: # BUY
#     class_weight_dict[0] *= additional_weight_multiplier
# if 1 in class_weight_dict: # SELL
#     class_weight_dict[1] *= additional_weight_multiplier
# # КОНЕЦ НОВОГО БЛОКА

# print(f"📊 Веса классов для обучения: {class_weight_dict}")
# ...



Перенесите вычисление class_weight_dict в train_xlstm_rl_system и передайте его в xlstm_model.train():

В функции train_xlstm_rl_system, после разделения данных на train/val/test, но перед вызовом xlstm_model.train(), добавьте вычисление class_weight_dict и передайте его в xlstm_model.train().

# В train_model.py, в функции train_xlstm_rl_system(...):
# ...
# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# =====================================================================
# НОВЫЙ БЛОК: ВЫЧИСЛЕНИЕ И ПЕРЕДАЧА ВЕСОВ КЛАССОВ
# =====================================================================
from sklearn.utils.class_weight import compute_class_weight
y_integers = np.argmax(y_train, axis=1)
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weight_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}

# Дополнительное усиление весов BUY/SELL
additional_weight_multiplier = 1.5
if 0 in class_weight_dict:
    class_weight_dict[0] *= additional_weight_multiplier
if 1 in class_weight_dict:
    class_weight_dict[1] *= additional_weight_multiplier

print(f"📊 Веса классов для обучения: {class_weight_dict}")
# =====================================================================
# КОНЕЦ НОВОГО БЛОКА
# =====================================================================

# ДОБАВЬТЕ: Принудительная очистка памяти
gc.collect()
tf.keras.backend.clear_session()

# ... (существующий код, включая создание xlstm_model) ...

# Обучение с улучшенными колбэками
history = xlstm_model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=16,
    class_weight=class_weight_dict, # <--- ДОБАВЛЕНО: передаем class_weight_dict
    custom_callbacks=[
        # ... (существующие колбэки) ...
    ]
)
# ...



Почему это исправит ошибку:
Теперь метод train в XLSTMRLModel будет ожидать аргумент class_weight и корректно передаст его в model.fit(). Мы также переместили логику вычисления class_weight_dict в train_xlstm_rl_system, чтобы она выполнялась один раз для всей системы обучения, а не внутри класса модели.
После этих изменений ошибка TypeError должна исчезнуть, и обучение должно продолжиться с использованием взвешивания классов, как мы и планировали.