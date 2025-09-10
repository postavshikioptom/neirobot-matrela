import json
import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class NotificationSystem:
    """
    Система уведомлений о важных событиях бота
    """
    
    def __init__(self, config_file='notification_config.json'):
        self.config = self._load_config(config_file)
        
    def _load_config(self, config_file):
        """Загружает конфигурацию уведомлений"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            # Конфигурация по умолчанию
            default_config = {
                "telegram": {
                    "enabled": False,
                    "bot_token": "",
                    "chat_id": ""
                },
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "email": "",
                    "password": "",
                    "to_email": ""
                },
                "webhook": {
                    "enabled": False,
                    "url": ""
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                
            print(f"📝 Создан файл конфигурации уведомлений: {config_file}")
            return default_config
    
    def send_trade_alert(self, symbol, action, price, pnl=None, reason=""):
        """Отправляет уведомление о сделке"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if action == "OPEN":
            message = f"🚀 [{timestamp}] Открыта позиция {symbol}\n💰 Цена: {price}\n📊 Причина: {reason}"
        else:
            pnl_emoji = "📈" if pnl > 0 else "📉"
            message = f"{pnl_emoji} [{timestamp}] Закрыта позиция {symbol}\n💰 Цена: {price}\n💵 PnL: {pnl:.2f}%\n📊 Причина: {reason}"
        
        self._send_notification(message, priority="normal")
    
    def send_system_alert(self, message, priority="high"):
        """Отправляет системное уведомление"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"⚠️ [{timestamp}] СИСТЕМА: {message}"
        self._send_notification(full_message, priority)
    
    def send_performance_report(self, daily_stats):
        """Отправляет отчет о производительности"""
        win_rate = (daily_stats['winning_trades'] / max(daily_stats['trades_closed'], 1)) * 100
        
        message = f"""📊 ДНЕВНОЙ ОТЧЕТ
🕐 Дата: {daily_stats['date']}
📈 Сделок: {daily_stats['trades_closed']}
💰 PnL: {daily_stats['total_pnl']:.2f}%
🎯 Win Rate: {win_rate:.1f}%
✅ VSA подтвержденных: {daily_stats['vsa_confirmed_trades']}"""
        
        self._send_notification(message, priority="low")
    
    def _send_notification(self, message, priority="normal"):
        """Отправляет уведомление через все активные каналы"""
        if self.config['telegram']['enabled']:
            self._send_telegram(message)
            
        if self.config['email']['enabled'] and priority in ["high", "critical"]:
            self._send_email(message)
            
        if self.config['webhook']['enabled']:
            self._send_webhook(message, priority)
    
    def _send_telegram(self, message):
        """Отправляет уведомление в Telegram"""
        try:
            bot_token = self.config['telegram']['bot_token']
            chat_id = self.config['telegram']['chat_id']
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            print(f"❌ Ошибка отправки Telegram: {e}")
    
    def _send_email(self, message):
        """Отправляет email уведомление"""
        try:
            config = self.config['email']
            
            msg = MIMEText(message)
            msg['Subject'] = 'Trading Bot Alert'
            msg['From'] = config['email']
            msg['To'] = config['to_email']
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['email'], config['password'])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"❌ Ошибка отправки email: {e}")
    
    def _send_webhook(self, message, priority):
        """Отправляет webhook уведомление"""
        try:
            url = self.config['webhook']['url']
            data = {
                'message': message,
                'priority': priority,
                'timestamp': datetime.now().isoformat()
            }
            
            requests.post(url, json=data, timeout=10)
        except Exception as e:
            print(f"❌ Ошибка отправки webhook: {e}")