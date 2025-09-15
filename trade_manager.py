import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
import time
# import logging # 🔥 УДАЛЕНО: Импорт logging
import json
from datetime import datetime
import os

class TradeManager:
    """
    Менеджер для управления торговлей на бирже Bybit
    """
    def __init__(self, api_key, api_secret, api_url, order_amount, symbol, leverage="2"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = api_url
        self.order_amount = order_amount
        self.symbol = symbol
        self.leverage = leverage
        
        # Инициализация API
        self.session = HTTP(
            testnet=(api_url == "https://api-testnet.bybit.com"),
            api_key=api_key,
            api_secret=api_secret
        )
        
        # 🔥 УДАЛЕНО: Инициализация логгера и файлового обработчика
        # self.logger = logging.getLogger('trade_manager')
        # self.logger.setLevel(logging.INFO)
        # if not self.logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     self.logger.addHandler(handler)
            
        #     file_handler = logging.FileHandler('trading.log')
        #     file_handler.setFormatter(formatter)
        #     self.logger.addHandler(file_handler)
        
        # Инициализация торгового журнала
        self.trade_log = []
        self.position = 0  # 0 - нет позиции, 1 - длинная позиция, -1 - короткая позиция
        
        # Устанавливаем плечо
        self._set_leverage()
    
    def _set_leverage(self):
        """
        Устанавливает плечо для торговли
        """
        try:
            response = self.session.set_leverage(
                category="linear",
                symbol=self.symbol,
                buyLeverage=self.leverage,
                sellLeverage=self.leverage
            )
            
            if response['retCode'] == 0:
                # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                print(f"Установлено плечо {self.leverage} для {self.symbol}")
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.warning -> print
                print(f"Не удалось установить плечо: {response['retMsg']}")
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при установке плеча: {e}")
    
    def get_current_price(self):
        """
        Получает текущую цену инструмента
        """
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=self.symbol
            )
            
            if response['retCode'] == 0:
                price = float(response['result']['list'][0]['lastPrice'])
                # 🔥 ИЗМЕНЕНО: self.logger.debug -> print
                print(f"Текущая цена {self.symbol}: {price}")
                return price
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.warning -> print
                print(f"Не удалось получить текущую цену: {response['retMsg']}")
                return None
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при получении текущей цены: {e}")
            return None
    
    def place_order(self, action):
        """
        Размещает ордер на бирже
        
        action: 0 - BUY, 1 - HOLD, 2 - SELL
        """
        if action == 1:  # HOLD - ничего не делаем
            return True
        
        try:
            current_price = self.get_current_price()
            
            if current_price is None:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print("Не удалось получить текущую цену для размещения ордера")
                return False
            
            # Определяем тип ордера и сторону
            if action == 0:  # BUY
                side = "Buy"
                if self.position == -1:  # Если у нас короткая позиция, закрываем её
                    # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                    print("Закрываем короткую позицию")
                    self._close_position()
            elif action == 2:  # SELL
                side = "Sell"
                if self.position == 1:  # Если у нас длинная позиция, закрываем её
                    # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                    print("Закрываем длинную позицию")
                    self._close_position()
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print(f"Неизвестное действие: {action}")
                return False
            
            # Вычисляем количество контрактов
            qty = self.order_amount / current_price
            
            # Размещаем рыночный ордер
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(round(qty, 4)),
                timeInForce="GTC"
            )
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                print(f"Размещен {side} ордер на {qty} {self.symbol} по рыночной цене. ID: {order_id}")
                
                # Обновляем позицию
                if action == 0:  # BUY
                    self.position = 1
                elif action == 2:  # SELL
                    self.position = -1
                
                # Записываем в журнал
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'BUY' if action == 0 else 'SELL',
                    'price': current_price,
                    'qty': qty,
                    'order_id': order_id
                })
                
                # Сохраняем журнал
                self._save_trade_log()
                
                return True
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print(f"Ошибка при размещении ордера: {response['retMsg']}")
                return False
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при размещении ордера: {e}")
            return False
    
    def _close_position(self):
        """
        Закрывает текущую позицию
        """
        try:
            # Получаем информацию о текущей позиции
            response = self.session.get_positions(
                category="linear",
                symbol=self.symbol
            )
            
            if response['retCode'] != 0:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print(f"Ошибка при получении информации о позиции: {response['retMsg']}")
                return False
            
            position_info = response['result']['list'][0]
            size = float(position_info['size'])
            
            if size == 0:
                # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                print("Нет открытой позиции для закрытия")
                self.position = 0
                return True
            
            # Определяем сторону для закрытия позиции
            side = "Sell" if position_info['side'] == "Buy" else "Buy"
            
            # Размещаем рыночный ордер для закрытия позиции
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(size),
                timeInForce="GTC",
                reduceOnly=True
            )
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                # 🔥 ИЗМЕНЕНО: self.logger.info -> print
                print(f"Закрыта позиция {self.symbol}. ID ордера: {order_id}")
                self.position = 0
                
                # Записываем в журнал
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'CLOSE',
                    'price': self.get_current_price(),
                    'qty': size,
                    'order_id': order_id
                })
                
                # Сохраняем журнал
                self._save_trade_log()
                
                return True
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.error -> print
                print(f"Ошибка при закрытии позиции: {response['retMsg']}")
                return False
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при закрытии позиции: {e}")
            return False
    
    def get_position_info(self):
        """
        Получает информацию о текущей позиции
        """
        try:
            response = self.session.get_positions(
                category="linear",
                symbol=self.symbol
            )
            
            if response['retCode'] == 0:
                position_info = response['result']['list'][0]
                return {
                    'symbol': position_info['symbol'],
                    'side': position_info['side'],
                    'size': float(position_info['size']),
                    'entry_price': float(position_info['entryPrice']),
                    'leverage': float(position_info['leverage']),
                    'unrealised_pnl': float(position_info['unrealisedPnl']),
                    'position_value': float(position_info['positionValue'])
                }
            else:
                # 🔥 ИЗМЕНЕНО: self.logger.warning -> print
                print(f"Не удалось получить информацию о позиции: {response['retMsg']}")
                return None
        
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при получении информации о позиции: {e}")
            return None
    
    def _save_trade_log(self):
        """
        Сохраняет журнал торговли в файл
        """
        try:
            with open('trade_log.json', 'w') as f:
                json.dump(self.trade_log, f, indent=2)
        except Exception as e:
            # 🔥 ИЗМЕНЕНО: self.logger.error -> print
            print(f"Ошибка при сохранении журнала торговли: {e}")