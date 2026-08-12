from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


DEFAULT_OLLAMA_URL = (
    "http://127.0.0.1:11434"
)

DEFAULT_MODEL = (
    "qwen3:4b"
)


ANALYST_SCHEMA = {
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


FORBIDDEN_ADVICE_PHRASES = (
    "almalısın",
    "satmalısın",
    "alım yap",
    "satış yap",
    "yatırım yap",
    "yatırım yapmalısın",
    "pozisyon aç",
    "pozisyon kapat",
    "beklemelisin",
    "hareket etmelisin",
    "dikkatli hareket et",
    "tercih etmelisin",
    "değerlendirmelisin",
    "kesin yükselecek",
    "kesin düşecek",
    "garantili",
    "garanti kazanç",
)


FORBIDDEN_ADVICE_PATTERNS = (
    r"\balım\s+fırsat",
    r"\bsatış\s+fırsat",
    r"\bsatım\s+fırsat",
    r"\byatırım\s+yap",
    r"\byatırım\s+için\s+uygun",
    r"\bpozisyon\s+aç",
    r"\bpozisyon\s+kapat",
    r"\bpozisyona\s+gir",
    r"\bişlem\s+aç",
    r"\bişlem\s+kapat",
    r"\balım\s+için\s+uygun",
    r"\bsatış\s+için\s+uygun",
    r"\bsatım\s+için\s+uygun",
    r"\bkâr\s+fırsat",
    r"\bkazanç\s+fırsat",
    r"\bfırsat\s+olabilir",
)



class LocalLLMAnalyst:
    """
    Local educational explanation layer.

    The LLM does not calculate indicators.
    It only explains structured output produced
    by MarketInterpreter.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = (
            model
            or os.getenv(
                "OLLAMA_MODEL",
                DEFAULT_MODEL,
            )
        )

        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                DEFAULT_OLLAMA_URL,
            )
        ).rstrip("/")

        self.timeout = timeout


    def _fallback(
        self,
        interpretation: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        important = [
            {
                "name":
                    item.get("name"),

                "direction":
                    item.get("direction"),

                "importance":
                    item.get("importance"),
            }
            for item
            in interpretation.get(
                "important",
                [],
            )[:3]
        ]

        low_relevance = [
            {
                "name":
                    item.get("name"),

                "direction":
                    item.get("direction"),

                "importance":
                    item.get("importance"),
            }
            for item
            in interpretation.get(
                "low_relevance",
                [],
            )[:3]
        ]

        conflicts = (
            interpretation.get(
                "conflicts",
                [],
            )
        )

        if conflicts:
            risk_note = (
                "Teknik göstergeler arasında "
                "yön uyuşmazlığı bulunuyor. "
                "Bu nedenle göstergeler birbirini "
                "tam olarak teyit etmiyor."
            )
        else:
            risk_note = (
                "Bu açıklama yalnızca mevcut "
                "teknik durumu özetler; gelecekteki "
                "fiyat hareketini tahmin etmez."
            )

        return {
            "available":
                False,

            "source":
                "deterministic_fallback",

            "model":
                self.model,

            "reason":
                reason,

            "summary":
                (
                    "Teknik göstergeler mevcut piyasa "
                    "durumunda birlikte değerlendirildi."
                ),

            "explanation":
                self._build_turkish_fallback_explanation(
                    interpretation
                ),

            "educational_note":
                risk_note,

            "important":
                important,

            "low_relevance":
                low_relevance,
        }


    @staticmethod
    def _build_turkish_fallback_explanation(
        interpretation: dict[str, Any],
    ) -> str:
        trend = (
            interpretation
            .get("trend", {})
            .get("direction", "NEUTRAL")
        )

        momentum = (
            interpretation
            .get("momentum", {})
            .get("direction", "NEUTRAL")
        )

        translations = {
            "BULLISH": "yukarı yönlü",
            "BEARISH": "aşağı yönlü",
            "NEUTRAL": "nötr",
        }

        trend_text = translations.get(
            trend,
            trend.lower(),
        )

        momentum_text = translations.get(
            momentum,
            momentum.lower(),
        )

        parts = [
            (
                "SMA trendi şu anda "
                + trend_text
                + "."
            ),
            (
                "MACD momentumu ise "
                + momentum_text
                + "."
            ),
        ]

        conflicts = interpretation.get(
            "conflicts",
            [],
        )

        if conflicts:
            parts.append(
                "Göstergeler arasında yön uyuşmazlığı bulunuyor."
            )

        return " ".join(parts)


    @staticmethod
    def _parse_json_content(
        content: str,
    ) -> dict[str, Any] | None:
        if not isinstance(
            content,
            str,
        ):
            return None

        original = content.strip()

        if not original:
            return None


        def try_parse(
            value: str,
        ) -> dict[str, Any] | None:
            try:
                parsed = json.loads(
                    value.strip()
                )

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                return None

            if not isinstance(
                parsed,
                dict,
            ):
                return None

            return parsed


        # 1. Ideal response:
        # pure JSON.
        parsed = try_parse(
            original
        )

        if parsed is not None:
            return parsed


        # 2. Some local models / tests can
        # contain escaped newline sequences.
        normalized = original.replace(
            "\\n",
            "\n",
        )


        # Try normalized content directly.
        parsed = try_parse(
            normalized
        )

        if parsed is not None:
            return parsed


        # 3. Markdown fenced JSON:
        #
        # ```json
        # {...}
        # ```
        fenced = normalized.strip()

        if fenced.startswith(
            "```"
        ):
            first_newline = (
                fenced.find(
                    "\n"
                )
            )

            if first_newline != -1:
                fenced = (
                    fenced[
                        first_newline + 1:
                    ]
                )

            else:
                # Handles rare one-line
                # ```json {...} ``` responses.
                if fenced.startswith(
                    "```json"
                ):
                    fenced = (
                        fenced[
                            len("```json"):
                        ]
                    )

                elif fenced.startswith(
                    "```"
                ):
                    fenced = (
                        fenced[
                            len("```"):
                        ]
                    )

            if fenced.rstrip().endswith(
                "```"
            ):
                fenced = (
                    fenced.rstrip()[:-3]
                )

            fenced = fenced.strip()

            parsed = try_parse(
                fenced
            )

            if parsed is not None:
                return parsed


        # 4. Safe recovery from surrounding
        # prose, e.g.
        #
        # "İstenen çıktı: {...}"
        #
        # Always use normalized ORIGINAL
        # text, not the modified fenced copy.
        start = normalized.find(
            "{"
        )

        end = normalized.rfind(
            "}"
        )

        if (
            start == -1
            or end == -1
            or end <= start
        ):
            return None

        candidate = normalized[
            start:end + 1
        ]

        return try_parse(
            candidate
        )


    @staticmethod
    def _find_advice_violation(
        result: dict[str, Any],
    ) -> str | None:
        combined = " ".join(
            [
                str(
                    result.get(
                        "summary",
                        "",
                    )
                ),

                str(
                    result.get(
                        "explanation",
                        "",
                    )
                ),

                str(
                    result.get(
                        "educational_note",
                        "",
                    )
                ),
            ]
        ).lower()

        for phrase in (
            FORBIDDEN_ADVICE_PHRASES
        ):
            if phrase in combined:
                return (
                    "phrase:"
                    + phrase
                )

        for pattern in (
            FORBIDDEN_ADVICE_PATTERNS
        ):
            if re.search(
                pattern,
                combined,
                flags=re.IGNORECASE,
            ):
                return (
                    "pattern:"
                    + pattern
                )

        return None


    @staticmethod
    def _contains_advice(
        result: dict[str, Any],
    ) -> bool:
        return (
            LocalLLMAnalyst
            ._find_advice_violation(
                result
            )
            is not None
        )

    def analyze(
        self,
        interpretation: dict[str, Any],
    ) -> dict[str, Any]:
        if not interpretation.get(
            "ready"
        ):
            return self._fallback(
                interpretation,
                "market_not_ready",
            )

        safe_input = {
            "market_state":
                interpretation.get(
                    "state"
                ),

            "heuristic_confidence":
                interpretation.get(
                    "confidence"
                ),

            "headline":
                interpretation.get(
                    "headline"
                ),

            "trend":
                interpretation.get(
                    "trend"
                ),

            "momentum":
                interpretation.get(
                    "momentum"
                ),

            "rsi":
                interpretation.get(
                    "rsi"
                ),

            "volatility":
                interpretation.get(
                    "volatility"
                ),

            "important":
                interpretation.get(
                    "important",
                    [],
                ),

            "low_relevance":
                interpretation.get(
                    "low_relevance",
                    [],
                ),

            "conflicts":
                interpretation.get(
                    "conflicts",
                    [],
                ),

            "technical_score":
                interpretation.get(
                    "technical_score"
                ),
        }

        system_prompt = """
Sen finansal piyasa grafiklerini yeni başlayan
kullanıcılara açıklayan eğitim amaçlı bir asistansın.

Sadece sana verilen JSON verisini açıkla.

Kurallar:
- Yeni finansal veri veya sayı uydurma.
- Gelecekteki fiyatı tahmin etme.
- Kullanıcıya al, sat, bekle, pozisyon aç veya
  yatırım yap şeklinde talimat verme.
- SELL ve BUY değerleri işlem tavsiyesi değildir;
  bunlar sadece teknik analiz motorunun etiketleridir.
- importance değerlerini kesinlikle değiştirme.
- Hangi göstergenin önemli olduğuna karar verme.
  Bu karar sana verilen veride zaten verilmiştir.
- "important" listesindeki göstergeler ana açıklamanın temelidir.
- "low_relevance" listesindeki göstergeler yalnızca arka plan bilgisidir.
- LOW importance olan veriyi önemliymiş gibi anlatma.
- LOW importance yalnızca mevcut piyasa anında
  düşük açıklayıcı öneme sahip demektir.
- LOW importance için "etkisizdir", "önemsizdir",
  "piyasayı etkilemez" veya benzeri kesin ifadeler kullanma.
- Aynı göstergeyi hem önemli hem önemsiz olarak tanımlama.
- Önemli listede tek gösterge varsa diğer göstergelerden teyit
  varmış gibi davranma.
- Çelişki varsa açıkça belirt.
- Confidence bir olasılık değildir.
- BUY ve SELL kelimelerini işlem tavsiyesi gibi kullanma.
- Kullanıcıya ne yapması gerektiğini söyleme.
- "alım", "satım", "yatırım", "pozisyon", "işlem fırsatı"
  veya "fırsat" dili kullanma.
- "alım eğilimi" yerine "yukarı yönlü teknik eğilim" de.
- "satış eğilimi" yerine "aşağı yönlü teknik eğilim" de.
- "dikkatli ol", "bekle", "al", "sat" gibi öneriler verme.
- Yalnızca mevcut göstergelerin ne anlattığını tarif et.
- Doğal ve düzgün Türkçe kullan.
- Anlamsız veya yapay kelimeler üretme.
- Teknik terimi kullandığında kısa şekilde açıkla.
- Gereksiz tekrar yapma.
- summary en fazla 2 kısa cümle olsun.
- explanation en fazla 3 kısa cümle olsun.
- educational_note yalnızca eğitim amaçlı bağlam versin.

Çıktıyı verilen JSON şemasına kesin olarak uydur.
""".strip()

        schema_text = (
            json.dumps(
                ANALYST_SCHEMA,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

        user_prompt = (
            "Aşağıdaki mevcut piyasa "
            "yorumunu açıkla. "
            "Yalnızca geçerli JSON döndür. "
            "Markdown kod bloğu kullanma. "
            "JSON öncesinde veya sonrasında "
            "başka metin yazma.\n\n"
            "MARKET DATA:\n"
            + json.dumps(
                safe_input,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
            + "\n\nJSON SCHEMA:\n"
            + schema_text
        )

        payload = {
            "model":
                self.model,

            "stream":
                False,

            "think":
                False,

            "format":
                ANALYST_SCHEMA,

            "options": {
                "temperature":
                    0,

                "num_predict":
                    600,
            },

            "messages": [
                {
                    "role":
                        "system",

                    "content":
                        system_prompt,
                },

                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
                },
            ],
        }

        request = urllib.request.Request(
            (
                self.base_url
                + "/api/chat"
            ),
            data=json.dumps(
                payload
            ).encode(
                "utf-8"
            ),
            headers={
                "Content-Type":
                    "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                api_result = (
                    json.load(
                        response
                    )
                )

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            return self._fallback(
                interpretation,
                (
                    "ollama_unavailable:"
                    + type(exc).__name__
                ),
            )

        content = (
            api_result
            .get(
                "message",
                {},
            )
            .get(
                "content",
                "",
            )
        )

        result = (
            self._parse_json_content(
                content
            )
        )

        if result is None:
            fallback = self._fallback(
                interpretation,
                "invalid_llm_json",
            )

            if (
                os.getenv(
                    "OLLAMA_DEBUG"
                )
                == "1"
            ):
                fallback["debug"] = {
                    "done":
                        api_result.get(
                            "done"
                        ),

                    "done_reason":
                        api_result.get(
                            "done_reason"
                        ),

                    "eval_count":
                        api_result.get(
                            "eval_count"
                        ),

                    "raw_content":
                        content[:1000],
                }

            return fallback

        required = {
            "summary",
            "explanation",
            "educational_note",
        }

        if not required.issubset(
            result
        ):
            return self._fallback(
                interpretation,
                "invalid_llm_schema",
            )

        advice_violation = (
            self._find_advice_violation(
                result
            )
        )

        if advice_violation:
            fallback = self._fallback(
                interpretation,
                "financial_advice_blocked",
            )

            if (
                os.getenv(
                    "OLLAMA_DEBUG"
                )
                == "1"
            ):
                fallback["debug"] = {
                    "violation":
                        advice_violation,

                    "llm_result":
                        result,
                }

            return fallback

        return {
            "available":
                True,

            "source":
                "ollama",

            "model":
                self.model,

            **result,

            "important": [
                {
                    "name":
                        item.get("name"),

                    "direction":
                        item.get("direction"),

                    "importance":
                        item.get("importance"),
                }
                for item
                in interpretation.get(
                    "important",
                    [],
                )[:3]
            ],

            "low_relevance": [
                {
                    "name":
                        item.get("name"),

                    "direction":
                        item.get("direction"),

                    "importance":
                        item.get("importance"),
                }
                for item
                in interpretation.get(
                    "low_relevance",
                    [],
                )[:3]
            ],
        }
