import tensorflow as tf
import numpy as np

class AntiOverfittingCallback(tf.keras.callbacks.Callback):
    """
    Колбэк для борьбы с переобучением
    """
    def __init__(self, patience=8, min_improvement=0.005):
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.wait = 0
        self.best_val_loss = np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        train_loss = logs.get('loss')
        
        # Проверяем на переобучение
        overfitting_ratio = val_loss / train_loss if train_loss > 0 else 1.0
        
        if overfitting_ratio > 1.3:  # Если val_loss > train_loss * 1.3
            print(f"\n⚠️ Обнаружено переобучение на эпохе {epoch+1}!")
            print(f"   Val_loss/Train_loss = {overfitting_ratio:.2f}")
            
            # Мы не будем напрямую менять LR, если он управляется ReduceLROnPlateau.
            # Вместо этого мы будем полагаться на ReduceLROnPlateau для снижения LR,
            # а наш колбэк будет только отслеживать overfitting и, возможно, останавливать обучение.

            # Если val_loss увеличивается, наш колбэк может остановить обучение,
            # но не будет пытаться изменить LR, чтобы избежать конфликтов.
        
        # Отслеживаем улучшение val_loss
        if val_loss < self.best_val_loss - self.min_improvement:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience and epoch > 20:  # Минимум 20 эпох
            print(f"\n🛑 Останавливаем обучение: нет улучшений {self.patience} эпох")
            self.model.stop_training = True