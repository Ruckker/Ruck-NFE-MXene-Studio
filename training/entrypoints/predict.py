# ==============================================================================
# 中文概述：对生产预测入口增加图缓存/全局特征语义兼容性检查。
# English overview: Guard production inference against incompatible graph/global-feature semantics.
# ==============================================================================

from nfe_model.predict_guard import main


if __name__ == "__main__":
    raise SystemExit(main())
