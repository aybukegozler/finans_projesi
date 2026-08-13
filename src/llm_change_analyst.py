from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib import error
from urllib import request


DEFAULT_OLLAMA_URL = (
    "http://127.0.0.1:11434"
)

DEFAULT_MODEL = "qwen3:4b"


CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
        },
        "explanation": {
            "type": "string",
        },
        "educational_note": {
            "type": "string",
        },
    },
    "required": [
        "summary",
        "explanation",
        "educational_note",
    ],
    "additionalProperties": False,
}


FORBIDDEN_PATTERNS = [
    r"\balım fırsat",
    r"\bsatış fırsat",
    r"\byatırım yap",
    r"\bpozisyon aç",
    r"\bpozisyon kapat",
    r"\bpozisyona gir",
    r"\bişlem aç",
    r"\bişlem kapat",
    r"\balmalısın",
    r"\bsatmalısın",
]


class LocalChangeAnalyst:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_URL",
                DEFAULT_OLLAMA_URL,
            )
        ).rstrip("/")

        self.model = (
            model
            or os.getenv(
                "OLLAMA_MODEL",
                DEFAULT_MODEL,
            )
        )

        self.timeout = timeout

    @staticmethod
    def _fallback(
        change: dict[str, Any],
        *,
        reason: str,
    ) -> dict[str, Any]:
        headline = change.get(
            "headline",
            (
                "Son analizden beri "
                "belirgin değişim yok."
            ),
        )

        changes = change.get(
            "changes",
            [],
        )

        if changes:
            readable = []

            for item in changes[:4]:
                label = item.get(
                    "label",
                    "Gösterge",
                )

                before = item.get(
                    "before",
                    "--",
                )

                after = item.get(
                    "after",
                    "--",
                )

                readable.append(
                    f"{label}: "
                    f"{before} → {after}"
                )

            explanation = (
                "Tespit edilen değişiklikler: "
                + "; ".join(readable)
                + "."
            )

        else:
            explanation = (
                "Ana teknik yapı önceki "
                "analize göre belirgin "
                "şekilde değişmedi."
            )

        return {
            "available": False,
            "source": "deterministic",
            "model": None,
            "reason": reason,
            "summary": headline,
            "explanation": explanation,
            "educational_note": (
                "Bu karşılaştırma yalnızca "
                "iki teknik analiz anını "
                "karşılaştırır ve yatırım "
                "tavsiyesi değildir."
            ),
        }

    @staticmethod
    def _parse_json(
        content: str,
    ) -> dict[str, Any] | None:
        cleaned = content.strip()

        if not cleaned:
            return None

        try:
            parsed = json.loads(
                cleaned
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

        fenced = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        fenced = re.sub(
            r"\s*```$",
            "",
            fenced,
        )

        try:
            parsed = json.loads(
                fenced
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if (
            start == -1
            or end == -1
            or end <= start
        ):
            return None

        try:
            parsed = json.loads(
                cleaned[
                    start:
                    end + 1
                ]
            )

        except json.JSONDecodeError:
            return None

        return (
            parsed
            if isinstance(
                parsed,
                dict,
            )
            else None
        )

    @staticmethod
    def _contains_advice(
        result: dict[str, Any],
    ) -> bool:
        text = " ".join(
            str(
                result.get(
                    key,
                    "",
                )
            )
            for key in (
                "summary",
                "explanation",
                "educational_note",
            )
        ).lower()

        return any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern
            in FORBIDDEN_PATTERNS
        )

    @staticmethod
    def _valid_result(
        result: dict[str, Any],
    ) -> bool:
        required = (
            "summary",
            "explanation",
            "educational_note",
        )

        return all(
            isinstance(
                result.get(key),
                str,
            )
            and result[key].strip()
            for key in required
        )

    def analyze(
        self,
        change: dict[str, Any],
        mode: str = "simple",
    ) -> dict[str, Any]:
        if not change.get(
            "ready"
        ):
            return self._fallback(
                change,
                reason="change_not_ready",
            )

        if not change.get(
            "meaningful"
        ):
            return {
                "available": True,
                "source": "deterministic",
                "model": None,
                "reason": None,
                "summary": change.get(
                    "headline",
                    (
                        "Belirgin bir "
                        "değişim yok."
                    ),
                ),
                "explanation": (
                    "Trend, momentum ve "
                    "diğer ana teknik yapı "
                    "önceki analize göre "
                    "belirgin biçimde "
                    "değişmedi."
                ),
                "educational_note": (
                    "Bir göstergenin kısa "
                    "süre değişmemesi, "
                    "gelecekte değişmeyeceği "
                    "anlamına gelmez."
                ),
            }

        if mode not in {
            "simple",
            "technical",
        }:
            mode = "simple"

        if mode == "simple":
            mode_prompt = """
Açıklama modu BASİT.

Finans bilgisi olmayan birine anlat.
İngilizce teknik jargon kullanma.
Gereksiz sayı verme.
Önce büyük resmi anlat.
Teknik terim geçerse kısa şekilde açıkla.
""".strip()

        else:
            mode_prompt = """
Açıklama modu TEKNİK.

Teknik terimleri kullanabilirsin.
Yön değişikliklerini ve skor farklarını açıkla.
Trend, momentum, RSI, volatilite ve
Bollinger değişimlerini gerektiğinde belirt.
""".strip()

        safe_change = {
            "headline":
                change.get(
                    "headline"
                ),

            "changes":
                change.get(
                    "changes",
                    [],
                ),

            "confidence":
                change.get(
                    "confidence",
                    {},
                ),

            "technical_score":
                change.get(
                    "technical_score",
                    {},
                ),

            "importance_changes":
                change.get(
                    "importance_changes",
                    {},
                ),

            "conflicts":
                change.get(
                    "conflicts",
                    {},
                ),

            "previous_state":
                change.get(
                    "previous_state"
                ),

            "current_state":
                change.get(
                    "current_state"
                ),
        }

        system_prompt = """
Sen teknik piyasa analizindeki iki farklı
zaman anı arasındaki DEĞİŞİKLİKLERİ
açıklayan eğitim amaçlı bir asistansın.

Sadece verilen JSON verisini kullan.

Yeni piyasa verisi uydurma.
Fiyat tahmini yapma.
Yatırım tavsiyesi verme.
Alım/satım önerme.
Gelecekte ne olacağını kesin olarak söyleme.

Önemli:
- Göstergeler fiyatı veya piyasayı
  "etkilemez"; piyasa verisinden hesaplanır.
- HIGH/MEDIUM/LOW önem seviyeleri
  piyasa üzerindeki etki değildir.
- Bunlar yalnızca mevcut teknik açıklamada
  hangi bilginin daha açıklayıcı olduğunu gösterir.
- Bir göstergenin değişmesi tek başına
  trend dönüşü anlamına gelmez.
- Çelişkiler varsa açıkça belirt.
- Değişmeyen şeyleri gereksiz yere uzatma.

summary:
En fazla 2 kısa cümle.

explanation:
En fazla 4 cümle.

educational_note:
1 kısa eğitim cümlesi.

Yanıt yalnızca istenen JSON şemasına
uygun olmalıdır.
""".strip()

        prompt = (
            system_prompt
            + "\n\n"
            + mode_prompt
            + "\n\nDEĞİŞİKLİK VERİSİ:\n"
            + json.dumps(
                safe_change,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": CHANGE_SCHEMA,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": 450,
            },
        }

        body = json.dumps(
            payload
        ).encode(
            "utf-8"
        )

        http_request = request.Request(
            (
                self.base_url
                + "/api/chat"
            ),
            data=body,
            headers={
                "Content-Type":
                    "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self.timeout,
            ) as response:
                raw = response.read()

        except (
            error.URLError,
            TimeoutError,
            OSError,
        ):
            return self._fallback(
                change,
                reason="ollama_unavailable",
            )

        try:
            response_data = json.loads(
                raw.decode(
                    "utf-8"
                )
            )

        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return self._fallback(
                change,
                reason="invalid_ollama_response",
            )

        content = (
            response_data.get(
                "message",
                {},
            ).get(
                "content",
                "",
            )
        )

        result = self._parse_json(
            content
        )

        if (
            result is None
            or not self._valid_result(
                result
            )
        ):
            return self._fallback(
                change,
                reason="invalid_json",
            )

        if self._contains_advice(
            result
        ):
            return self._fallback(
                change,
                reason="financial_advice_blocked",
            )

        return {
            "available": True,
            "source": "ollama",
            "model": self.model,
            "reason": None,
            "mode": mode,
            **result,
        }
