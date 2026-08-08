# ==============================================================================
# 中文概述：生产预测入口固定使用最终 v2.4 pair-symmetric 图契约与完整 checkpoint 自证。
# English overview: Production inference uses the final v2.4 pair-symmetric graph contract and checkpoint self-audit.
# ==============================================================================

from nfe_model.pair_symmetric_graph import install_pair_symmetric_graph_contract

# Install before importing predict_formal/predict_guard because those modules
# capture graph/cache/data-implementation constants at import time.
install_pair_symmetric_graph_contract()

from nfe_model.predict_formal import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
