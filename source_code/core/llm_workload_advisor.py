import json
import os
import ssl
from typing import Dict, Optional, Tuple
from urllib import error, request

from core.traffic_generator import WORKLOAD_CLASSES

try:
    import certifi
except Exception:
    certifi = None

PATTERN_TO_WORKLOAD = {
    "daily_business": "business",
    "weekly_peak": "business",
    "night_batch": "data_batch",
    "flash_crowd": "social",
    "seasonal_wave": "streaming",
    "steady_state": "api_gateway",
    "sawtooth": "api_gateway",
    "random_walk": "social",
    "gradual_ramp": "business",
    "cascading_fail": "api_gateway",
    "maintenance": "data_batch",
    "campaign_burst": "social",
}


class LLMWorkloadAdvisor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        endpoint: str = "https://api.openai.com/v1/responses",
        timeout_sec: float = 20.0,
    ):
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY", "")
            or os.getenv("GEMINI_API_KEY", "")
        ).strip()
        self.model = model
        self.endpoint = endpoint
        self.timeout_sec = float(timeout_sec)
        self.allowed = set(WORKLOAD_CLASSES)
        self.ssl_context = self._build_ssl_context()

        self.calls_total = 0
        self.calls_success = 0
        self.calls_failed = 0
        self.last_error: Optional[str] = None

    @property
    def online(self) -> bool:
        return bool(self.api_key)

    def _build_ssl_context(self):
        if certifi is not None:
            return ssl.create_default_context(cafile=certifi.where())
        return ssl.create_default_context()

    def suggest_workload(self, context: Dict) -> Dict[str, Optional[str]]:
        self.calls_total += 1

        if not self.online:
            return {
                "workload": None,
                "source": "fallback",
                "confidence": None,
                "raw": None,
                "error": "API key missing",
            }

        try:
            raw_text = self._call_llm_api(context)
            wl, conf = self._parse_workload(raw_text)
            if wl is None:
                self.calls_failed += 1
                preview = (raw_text or "").replace("\n", " ").strip()
                self.last_error = (
                    "LLM output could not be mapped to workload class"
                    + (f" | raw='{preview[:200]}'" if preview else "")
                )
                return {
                    "workload": None,
                    "source": "fallback",
                    "confidence": conf,
                    "raw": raw_text,
                    "error": self.last_error,
                }

            self.calls_success += 1
            return {
                "workload": wl,
                "source": "llm",
                "confidence": conf,
                "raw": raw_text,
                "error": None,
            }
        except Exception as exc:
            self.calls_failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {
                "workload": None,
                "source": "fallback",
                "confidence": None,
                "raw": None,
                "error": self.last_error,
            }

    def report(self) -> Dict[str, Optional[str]]:
        return {
            "online": self.online,
            "model": self.model,
            "endpoint": self.endpoint,
            "calls_total": self.calls_total,
            "calls_success": self.calls_success,
            "calls_failed": self.calls_failed,
            "last_error": self.last_error,
        }

    def _call_llm_api(self, context: Dict) -> str:
        ep = self.endpoint.lower()
        if "/chat/completions" in ep:
            return self._call_chat_completions_api(context)
        return self._call_responses_api(context)

    def _build_prompts(self, context: Dict) -> Tuple[str, str]:
        system_msg = (
            "You are a datacenter workload classifier. "
            "Choose exactly one workload label from this list and return only that label text, nothing else: "
            + ", ".join(WORKLOAD_CLASSES)
            + "."
        )
        user_msg = (
            "Telemetry context (JSON):\n"
            + json.dumps(context, ensure_ascii=False)
            + "\nReturn one label now."
        )
        return system_msg, user_msg

    def _call_responses_api(self, context: Dict) -> str:
        system_msg, user_msg = self._build_prompts(context)
        payload = {
            "model": self.model,
            "instructions": system_msg,
            "input": user_msg,
            "max_output_tokens": 140,
        }
        parsed = self._post_json(payload)
        text = parsed.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        extracted = self._extract_output_text(parsed)
        if extracted:
            return extracted
        raise RuntimeError("No output_text in Responses API payload")

    def _call_chat_completions_api(self, context: Dict) -> str:
        system_msg, user_msg = self._build_prompts(context)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 140,
            "temperature": 0,
        }
        parsed = self._post_json(payload)
        txt = self._extract_chat_completion_text(parsed)
        if txt:
            return txt
        raise RuntimeError("No choices[0].message.content in chat/completions payload")

    def _post_json(self, payload: Dict) -> Dict:
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_sec, context=self.ssl_context) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e

        parsed = json.loads(body)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "error" in parsed[0]:
            raise RuntimeError(f"API error: {parsed[0]['error']}")
        if isinstance(parsed, dict) and parsed.get("error"):
            raise RuntimeError(f"API error: {parsed['error']}")
        return parsed

    @staticmethod
    def _extract_output_text(payload: Dict) -> str:
        parts = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    txt = c.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_chat_completion_text(payload: Dict) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    txt = block.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
            return "\n".join(parts).strip()
        return ""

    def _parse_workload(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        conf: Optional[str] = None
        wl_raw: Optional[str] = None
        text = (text or "").strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                wl_raw = str(obj.get("workload", "")).strip()
                conf_val = obj.get("confidence")
                if conf_val is not None:
                    conf = str(conf_val)
        except json.JSONDecodeError:
            # Some models prepend prose and then embed a JSON block.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    obj = json.loads(text[start : end + 1])
                    if isinstance(obj, dict):
                        wl_raw = str(obj.get("workload", "")).strip()
                        conf_val = obj.get("confidence")
                        if conf_val is not None:
                            conf = str(conf_val)
                except Exception:
                    pass

            if not wl_raw:
                lower = text.lower()
                for label in WORKLOAD_CLASSES:
                    if label in lower:
                        wl_raw = label
                        break
                if not wl_raw:
                    for pat, mapped in PATTERN_TO_WORKLOAD.items():
                        if pat in lower:
                            wl_raw = mapped
                            break

        if not wl_raw:
            return None, conf

        wl = wl_raw.lower().strip().replace(" ", "_").replace("-", "_")
        if wl in self.allowed:
            return wl, conf
        if wl in PATTERN_TO_WORKLOAD:
            mapped = PATTERN_TO_WORKLOAD[wl]
            if mapped in self.allowed:
                return mapped, conf
        return None, conf
