# ==============================================================================
# 中文概述：验证桌面生成任务的分阶段进度上报与两次尝试映射。
# English overview: Verify staged desktop generation progress and retry mapping.
#
# 中文输入：模拟的核心流水线百分比与回调消息。
# English inputs: Simulated core-pipeline percentages and callback messages.
# 中文输出：单调、有限且边界明确的 unittest 断言。
# English outputs: Unittest assertions for monotonicity, bounds, and callbacks.
#
# Author: Ruck
# Generated: 2026-07-30 09:10:00 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import unittest

from app.windows.nfe_mxene_studio.backend import (
    scale_generation_progress,
    schedule_callback_with_value,
    summarize_generation_attempts,
)
from nfe_model import strict_generation


class GenerationProgressTest(unittest.TestCase):
    """中文：覆盖 GUI 依赖的纯进度接口。 English: Cover GUI progress APIs."""

    def test_two_attempt_progress_is_monotonic(self) -> None:
        first = [
            scale_generation_progress(0, value)
            for value in (0, 10, 50, 100)
        ]
        second = [
            scale_generation_progress(1, value)
            for value in (0, 10, 50, 100)
        ]
        values = [float(value) for value in first + second]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[0], 3.0)
        self.assertEqual(values[-1], 96.0)
        self.assertIsNone(scale_generation_progress(0, None))

    def test_strict_pipeline_callback_can_be_restored(self) -> None:
        events: list[tuple[str, float | None]] = []
        callback = lambda message, percent: events.append((message, percent))
        previous = strict_generation.set_progress_callback(callback)
        try:
            strict_generation.report_progress("测试阶段", 42.0)
        finally:
            restored = strict_generation.set_progress_callback(previous)
        self.assertIs(restored, callback)
        self.assertEqual(events, [("测试阶段", 42.0)])

    def test_deferred_exception_is_bound_before_except_cleanup(self) -> None:
        callbacks = []
        received = []

        def schedule(_delay, callback):
            callbacks.append(callback)

        try:
            raise RuntimeError("expected failure")
        except RuntimeError as exc:
            schedule_callback_with_value(schedule, received.append, exc)

        self.assertEqual(len(callbacks), 1)
        callbacks[0]()
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], RuntimeError)
        self.assertEqual(str(received[0]), "expected failure")

    def test_failure_summary_includes_rejections_and_probabilities(self) -> None:
        summary = summarize_generation_attempts(
            [
                {
                    "attempt": 2,
                    "exported": 0,
                    "rejection_reasons": {
                        "target_label_mismatch": 11,
                        "matches_training_structure": 2,
                    },
                    "prediction_diagnostics": {
                        "target_probability_max": 0.42,
                        "predicted_label_counts": {
                            "medium": 10,
                            "low": 3,
                        },
                    },
                }
            ]
        )
        self.assertIn("target_label_mismatch=11", summary)
        self.assertIn("目标概率最高=0.420", summary)
        self.assertIn("low:3", summary)

    def test_failure_summary_reports_exact_skeleton_support(self) -> None:
        """An unsupported target class must be visible, never silently relaxed."""
        summary = summarize_generation_attempts(
            [
                {
                    "attempt": 1,
                    "exported": 0,
                    "rejection_reasons": {"target_label_mismatch": 9},
                    "skeleton_reference_support": {
                        "skeleton": "Nb-C-Nb",
                        "exact_count": 79,
                        "label_counts": {"medium": 62, "high": 17},
                        "score_min": 0.4834409654,
                        "score_max": 0.9101055861,
                    },
                }
            ]
        )
        self.assertIn("Nb-C-Nb共79项", summary)
        self.assertIn("low:0/medium:62/high:17", summary)
        self.assertIn("0.483–0.910", summary)


if __name__ == "__main__":
    unittest.main()
