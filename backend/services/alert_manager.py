import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)


class AlertManager:
    """
    告警通知管理器。支持钉钉、飞书、邮件三种通知方式。
    关机失败时自动调用。
    """

    def __init__(self, dingtalk_webhook: Optional[str] = None,
                       feishu_webhook: Optional[str] = None,
                       email_smtp: Optional[str] = None,
                       email_to: Optional[str] = None,
                       email_smtp_host: Optional[str] = None,
                       email_smtp_port: Optional[int] = None,
                       email_smtp_user: Optional[str] = None,
                       email_smtp_password: Optional[str] = None,
                       email_from: Optional[str] = None):
        self.dingtalk_webhook = dingtalk_webhook
        self.feishu_webhook = feishu_webhook
        self.email_smtp = email_smtp
        self.email_to = email_to
        self.email_smtp_host = email_smtp_host
        self.email_smtp_port = email_smtp_port
        self.email_smtp_user = email_smtp_user
        self.email_smtp_password = email_smtp_password
        self.email_from = email_from

    def send(self, title: str, detail: str):
        """发送告警，失败不影响主流程"""
        if self.dingtalk_webhook:
            self._send_dingtalk(title, detail)
        if self.feishu_webhook:
            self._send_feishu(title, detail)
        if self.email_smtp and self.email_to:
            self._send_email(title, detail)

    def _send_dingtalk(self, title: str, detail: str):
        try:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title,
                    "text": f"## {title}\n\n{detail}\n\n> 请立即处理！",
                },
            }
            httpx.post(self.dingtalk_webhook, json=payload, timeout=10)
        except Exception as e:
            logger.warning("[AlertManager] 钉钉通知发送失败: webhook=%s error=%s", self.dingtalk_webhook[:20] + "...", e)

    def _send_feishu(self, title: str, detail: str):
        try:
            payload = {
                "msg_type": "text",
                "content": {"text": f"{title}\n\n{detail}"},
            }
            httpx.post(self.feishu_webhook, json=payload, timeout=10)
        except Exception as e:
            logger.warning("[AlertManager] 飞书通知发送失败: webhook=%s error=%s", self.feishu_webhook[:20] + "...", e)

    def _send_email(self, title: str, detail: str):
        import smtplib
        from email.mime.text import MIMEText
        try:
            msg = MIMEText(detail, "plain", "utf-8")
            msg["Subject"] = title
            msg["From"] = self.email_from or self.email_smtp
            msg["To"] = self.email_to

            smtp_host = self.email_smtp_host or "smtp.gmail.com"
            smtp_port = self.email_smtp_port or 587
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                if self.email_smtp_user and self.email_smtp_password:
                    server.login(self.email_smtp_user, self.email_smtp_password)
                server.send_message(msg)
        except Exception as e:
            logger.warning("[AlertManager] 邮件通知发送失败: to=%s error=%s", self.email_to, e)
