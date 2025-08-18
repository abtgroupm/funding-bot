import json
import time
import threading
import urllib.parse
import urllib.request
from typing import Callable, Dict, Optional, Set, Any

class TelegramManager:
    def __init__(self, token: str, allowed_chat_ids: Set[int], default_chat_id: Optional[int] = None):
        self.base = f"https://api.telegram.org/bot{token}"
        self.allowed = set(allowed_chat_ids or [])
        self.default_chat_id = default_chat_id
        self._offset = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.handlers: Dict[str, Callable[[str, int], str]] = {}

    def on(self, cmd: str, handler: Callable[[str, int], str]):
        self.handlers[cmd] = handler

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="tg-long-poll", daemon=True)
        self._thread.start()

    def stop(self):
        # yêu cầu dừng
        self._running = False
        # "poke" 1 request ngắn để phá long-poll đang chờ
        try:
            self._api("getUpdates", {"timeout": 0, "offset": self._offset + 1})
        except Exception:
            pass
        # chờ thread kết thúc tối đa 2s (không block vô hạn)
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass

    def _api(self, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # serialize reply_markup if dict/list
        payload: Dict[str, Any] = {}
        for k, v in (data or {}).items():
            if k == "reply_markup" and isinstance(v, (dict, list)):
                payload[k] = json.dumps(v)
            else:
                payload[k] = v
        body = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(self.base + "/" + method, data=body)
        with urllib.request.urlopen(req, timeout=35) as r:
            return json.loads(r.read().decode("utf-8"))

    def reply_keyboard(self, rows, resize=True, one_time=False) -> Dict[str, Any]:
        # rows: List[List[str]]
        return {
            "keyboard": [[{"text": b} for b in row] for row in rows],
            "resize_keyboard": bool(resize),
            "one_time_keyboard": bool(one_time),
        }

    def send(self, text: str, chat_id: Optional[int] = None, keyboard: Optional[Any] = None):
        cid = chat_id or self.default_chat_id
        if not cid:
            return
        payload: Dict[str, Any] = {"chat_id": cid, "text": text}
        if keyboard:
            # keyboard có thể là list rows hoặc dict reply_markup
            payload["reply_markup"] = self.reply_keyboard(keyboard) if isinstance(keyboard, list) else keyboard
        try:
            self._api("sendMessage", payload)
        except Exception:
            pass

    def _loop(self):
        while self._running:
            try:
                resp = self._api("getUpdates", {"timeout": 30, "offset": self._offset + 1})
                for upd in resp.get("result", []):
                    self._offset = max(self._offset, int(upd.get("update_id", 0)))
                    msg = upd.get("message") or upd.get("edited_message") or {}
                    text = (msg.get("text") or "").strip()
                    chat = msg.get("chat") or {}
                    cid = int(chat.get("id", 0))
                    if not text or not cid:
                        continue
                    if self.allowed and cid not in self.allowed:
                        try:
                            self._api("sendMessage", {"chat_id": cid, "text": "Unauthorized chat."})
                        except Exception:
                            pass
                        continue
                    parts = text.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""
                    handler = self.handlers.get(cmd) or self.handlers.get("default")
                    if handler:
                        try:
                            reply = handler(arg, cid)
                        except Exception as e:
                            reply = f"Error handling {cmd}: {e}"
                        if reply:
                            self.send(reply, cid)
            except Exception:
                time.sleep(2)