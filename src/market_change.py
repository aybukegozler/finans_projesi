from __future__ import annotations

from typing import Any


class MarketChangeDetector:
    VERSION = "change-v1"

    @staticmethod
    def _nested(
        data: dict[str, Any],
        *keys: str,
    ) -> Any:
        current: Any = data

        for key in keys:
            if not isinstance(
                current,
                dict,
            ):
                return None

            current = current.get(key)

        return current

    @staticmethod
    def _number(
        value: Any,
    ) -> float | None:
        try:
            if value is None:
                return None

            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _confidence_score(
        cls,
        interpretation: dict[str, Any],
    ) -> float | None:
        confidence = interpretation.get(
            "confidence"
        )

        if isinstance(
            confidence,
            dict,
        ):
            for key in (
                "score",
                "value",
                "confidence",
            ):
                value = cls._number(
                    confidence.get(key)
                )

                if value is not None:
                    return value

            return None

        return cls._number(
            confidence
        )

    @staticmethod
    def _factor_names(
        factors: Any,
    ) -> set[str]:
        if not isinstance(
            factors,
            list,
        ):
            return set()

        names: set[str] = set()

        for factor in factors:
            if not isinstance(
                factor,
                dict,
            ):
                continue

            name = factor.get(
                "name"
            )

            if name:
                names.add(
                    str(name)
                )

        return names

    @staticmethod
    def _conflicts(
        interpretation: dict[str, Any],
    ) -> set[str]:
        conflicts = interpretation.get(
            "conflicts"
        )

        if not isinstance(
            conflicts,
            list,
        ):
            return set()

        return {
            str(item)
            for item in conflicts
            if item
        }

    @staticmethod
    def _add_change(
        changes: list[dict[str, Any]],
        *,
        category: str,
        label: str,
        before: Any,
        after: Any,
    ) -> None:
        if (
            before is None
            or after is None
            or before == after
        ):
            return

        changes.append(
            {
                "category": category,
                "label": label,
                "before": before,
                "after": after,
            }
        )

    def compare(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            not previous.get("ready")
            or not current.get("ready")
        ):
            return {
                "version": self.VERSION,
                "ready": False,
                "meaningful": False,
                "headline": (
                    "Karşılaştırma için iki hazır "
                    "analiz gerekiyor."
                ),
                "changes": [],
            }

        changes: list[
            dict[str, Any]
        ] = []

        self._add_change(
            changes,
            category="market",
            label="Market State",
            before=previous.get(
                "state"
            ),
            after=current.get(
                "state"
            ),
        )

        self._add_change(
            changes,
            category="trend",
            label="SMA Trend",
            before=self._nested(
                previous,
                "trend",
                "direction",
            ),
            after=self._nested(
                current,
                "trend",
                "direction",
            ),
        )

        self._add_change(
            changes,
            category="momentum",
            label="MACD Momentum",
            before=self._nested(
                previous,
                "momentum",
                "direction",
            ),
            after=self._nested(
                current,
                "momentum",
                "direction",
            ),
        )

        self._add_change(
            changes,
            category="rsi",
            label="RSI",
            before=(
                self._nested(
                    previous,
                    "rsi",
                    "state",
                )
                or self._nested(
                    previous,
                    "rsi",
                    "direction",
                )
            ),
            after=(
                self._nested(
                    current,
                    "rsi",
                    "state",
                )
                or self._nested(
                    current,
                    "rsi",
                    "direction",
                )
            ),
        )

        self._add_change(
            changes,
            category="volatility",
            label="Volatility",
            before=self._nested(
                previous,
                "volatility",
                "state",
            ),
            after=self._nested(
                current,
                "volatility",
                "state",
            ),
        )

        self._add_change(
            changes,
            category="bollinger",
            label="Bollinger Position",
            before=(
                self._nested(
                    previous,
                    "bollinger",
                    "position",
                )
                or self._nested(
                    previous,
                    "bollinger",
                    "state",
                )
            ),
            after=(
                self._nested(
                    current,
                    "bollinger",
                    "position",
                )
                or self._nested(
                    current,
                    "bollinger",
                    "state",
                )
            ),
        )

        self._add_change(
            changes,
            category="signal",
            label="Live Signal",
            before=previous.get(
                "live_signal"
            ),
            after=current.get(
                "live_signal"
            ),
        )

        previous_confidence = (
            self._confidence_score(
                previous
            )
        )

        current_confidence = (
            self._confidence_score(
                current
            )
        )

        confidence_delta = None

        if (
            previous_confidence
            is not None
            and current_confidence
            is not None
        ):
            confidence_delta = round(
                current_confidence
                - previous_confidence,
                2,
            )

        previous_score = self._number(
            previous.get(
                "technical_score"
            )
        )

        current_score = self._number(
            current.get(
                "technical_score"
            )
        )

        technical_score_delta = None

        if (
            previous_score
            is not None
            and current_score
            is not None
        ):
            technical_score_delta = round(
                current_score
                - previous_score,
                2,
            )

        previous_important = (
            self._factor_names(
                previous.get(
                    "important"
                )
            )
        )

        current_important = (
            self._factor_names(
                current.get(
                    "important"
                )
            )
        )

        previous_low = (
            self._factor_names(
                previous.get(
                    "low_relevance"
                )
            )
        )

        current_low = (
            self._factor_names(
                current.get(
                    "low_relevance"
                )
            )
        )

        became_important = sorted(
            current_important
            - previous_important
        )

        became_low_relevance = sorted(
            current_low
            - previous_low
        )

        previous_conflicts = (
            self._conflicts(
                previous
            )
        )

        current_conflicts = (
            self._conflicts(
                current
            )
        )

        added_conflicts = sorted(
            current_conflicts
            - previous_conflicts
        )

        resolved_conflicts = sorted(
            previous_conflicts
            - current_conflicts
        )

        meaningful_numeric_change = (
            (
                confidence_delta
                is not None
                and abs(
                    confidence_delta
                ) >= 5
            )
            or (
                technical_score_delta
                is not None
                and technical_score_delta
                != 0
            )
        )

        meaningful = bool(
            changes
            or became_important
            or became_low_relevance
            or added_conflicts
            or resolved_conflicts
            or meaningful_numeric_change
        )

        if any(
            item["category"]
            == "market"
            for item in changes
        ):
            headline = (
                "Genel piyasa durumu "
                "değişti."
            )

        elif changes:
            headline = (
                "Bazı teknik bileşenler "
                "yön değiştirdi."
            )

        elif (
            became_important
            or became_low_relevance
        ):
            headline = (
                "Göstergelerin açıklamadaki "
                "önem sırası değişti."
            )

        elif meaningful_numeric_change:
            headline = (
                "Ana yapı aynı, ancak "
                "teknik skorlar değişti."
            )

        else:
            headline = (
                "Son analizden beri belirgin "
                "bir yapısal değişim yok."
            )

        return {
            "version": self.VERSION,
            "ready": True,
            "meaningful": meaningful,
            "headline": headline,

            "previous_state":
                previous.get(
                    "state"
                ),

            "current_state":
                current.get(
                    "state"
                ),

            "changes":
                changes,

            "confidence":
                {
                    "previous":
                        previous_confidence,

                    "current":
                        current_confidence,

                    "delta":
                        confidence_delta,
                },

            "technical_score":
                {
                    "previous":
                        previous_score,

                    "current":
                        current_score,

                    "delta":
                        technical_score_delta,
                },

            "importance_changes":
                {
                    "became_important":
                        became_important,

                    "became_low_relevance":
                        became_low_relevance,
                },

            "conflicts":
                {
                    "added":
                        added_conflicts,

                    "resolved":
                        resolved_conflicts,
                },

            "current_important":
                current.get(
                    "important",
                    [],
                ),

            "current_low_relevance":
                current.get(
                    "low_relevance",
                    [],
                ),
        }
