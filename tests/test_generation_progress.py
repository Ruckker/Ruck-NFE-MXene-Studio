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

from app.windows.nfe_mxene_studio.backend import scale_generation_progress
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


if __name__ == "__main__":
    unittest.main()
