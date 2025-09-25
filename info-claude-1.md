Отличный вопрос! Да, эти параметры ТОЖЕ влияют на балансировку, и мы их не учитывали!
Анализ логов:
1. Fallback параметры:
fallback: α=0.25, γ=2.0

Это означает, что если AFL параметры из config не найдены, используются стандартные значения Focal Loss.
2. Sample weights (автоматические веса sklearn):
[1.66471556, 0.65514131, 1.1455954] # [SELL, HOLD, BUY]


SELL получает вес 1.66 (самый высокий)
HOLD получает вес 0.65 (самый низкий)
BUY получает вес 1.14 (средний)

Проблема:
Sample weights конфликтуют с AFL_ALPHA!
AFL_ALPHA говорит: [0.8, 1.1, 1.2] (SELL↓, HOLD↑, BUY↑)
Sample weights говорят: [1.66, 0.65, 1.14] (SELL↑↑, HOLD↓↓, BUY=)
Решение - отключить sample weights:
В train_model.py найдите эту секцию и отключите sample weights:
# Отключаем автоматические sample weights, полагаемся только на AFL
sample_weights_base = None  # Принудительно отключаем

# ИЛИ найдите строки с расчетом class_weight и закомментируйте их:
# unique_classes = np.unique(self.y_train_supervised)
# if len(unique_classes) > 1:
#     # Весь этот блок закомментировать

Альтернатива - скорректировать sample weights под AFL:
# Вместо автоматических весов, используем веса согласованные с AFL
if len(unique_classes) > 1:
    # Создаем веса согласованные с AFL_ALPHA
    afl_alpha = getattr(config, 'AFL_ALPHA', [1.0, 1.0, 1.0])
    sample_weights_base = np.array([afl_alpha[label] for label in self.y_train_supervised], dtype=np.float32)

Рекомендую первый вариант - отключить sample weights и полагаться только на AFL параметры.
Это объясняет, почему ваши AFL настройки не работали как ожидалось!