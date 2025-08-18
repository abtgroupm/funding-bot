import json, time, threading, urllib.parse, urllib.request
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
        # yêu cầu dừng + phá long-poll và join ngắn
        self._running = False
        try:
            self._api("getUpdates", {"timeout": 0, "offset": self._offset + 1})
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass

    def _api(self, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
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

    def send(self, text: str, chat_id: Optional[int] = None, keyboard: Optional[Any] = None):
        cid = chat_id or self.default_chat_id
        if not cid:
            return
        data: Dict[str, Any] = {"chat_id": cid, "text": text}
        if keyboard:
            data["reply_markup"] = keyboard
        try:
            self._api("sendMessage", data)
        except Exception:
            pass

    def _normalize_command(self, text: str) -> str:
        """Chuẩn hóa lệnh, bỏ @ và xử lý đúng cách."""
        if not text:
            return ""
        parts = text.split()
        cmd = parts[0]
        # Bỏ @botname nếu có
        if '@' in cmd:
            cmd = cmd.split('@')[0]
        return cmd.lower()

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
                    
                    # Xử lý command entities nếu có
                    entities = msg.get("entities", [])
                    cmd_entity = None
                    for e in entities:
                        if e.get("type") == "bot_command" and e.get("offset") == 0:
                            cmd_entity = text[0:e.get("length", 0)].lower()
                            break
                    
                    # Tách command và argument
                    parts = text.split(maxsplit=1)
                    raw_cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""
                    
                    # Ưu tiên cmd_entity nếu có
                    cmd = cmd_entity or raw_cmd
                    
                    # Thử cả với và không có dấu /
                    handlers_to_try = [cmd]
                    if cmd.startswith('/'):
                        handlers_to_try.append(cmd[1:])  # Không có /
                    else:
                        handlers_to_try.append('/' + cmd)  # Có /
                        
                    # Tìm handler phù hợp
                    handler = None
                    for try_cmd in handlers_to_try:
                        if try_cmd in self.handlers:
                            handler = self.handlers[try_cmd]
                            break
                    
                    if not handler:
                        handler = self.handlers.get("default")
                        
                    if handler:
                        try:
                            reply = handler(arg, cid)
                        except Exception as e:
                            reply = f"Error handling {cmd}: {e}"
                        if reply:
                            self.send(reply, cid)
            except Exception as e:
                print(f"Telegram loop error: {e}")
                time.sleep(2)