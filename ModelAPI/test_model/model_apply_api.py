# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import pickle
import warnings
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Suppress pandas fragmentation warnings from wide feature generation.
warnings.filterwarnings("ignore", category=PerformanceWarning)

DEFAULT_HORIZONS = (3, 7, 15, 30)
BUY_THRESHOLD = 0.90
AVOID_THRESHOLD = 0.10
TOP_N_LATEST = 10
MIN_LOOKBACK_TRADING_DAYS = 220
DEFAULT_SYMBOL_LOOKBACK_MONTHS = 24
DEFAULT_POPULAR_SYMBOLS = 10

CYCLE_K_VALUES = (3, 5, 7, 10)
CYCLE_N_VALUES = (5, 10, 20)
CYCLE_BASE_EDGES = (-0.08, -0.04, -0.01, 0.01, 0.04, 0.08)

NUMERIC_COLUMNS = [
    "LS_GiaDieuChinh",
    "LS_GiaDongCua",
    "LS_GiaMoCua",
    "LS_GiaCaoNhat",
    "LS_GiaThapNhat",
    "LS_KhoiLuongKhopLenh",
    "LS_GiaTriKhopLenh",
    "LS_KLThoaThuan",
    "LS_GtThoaThuan",
    "DL_SoLenhMua",
    "DL_KLDatMua",
    "DL_KLTB1LenhMua",
    "DL_SoLenhDatBan",
    "DL_KLDatBan",
    "DL_KLTB1LenhBan",
    "DL_ChenhLechKL",
    "KN_KLGDRong",
    "KN_GTDGRong",
    "KN_KLMua",
    "KN_GtMua",
    "KN_KLBan",
    "KN_GtBan",
    "KN_RoomConLai",
    "KN_DangSoHuu",
    "TD_KLcpMua",
    "TD_KlcpBan",
    "TD_GtMua",
    "TD_GtBan",
]

DROP_COLUMNS = [
    "LS_ThayDoi",
    "DL_ThayDoi",
    "KN_ThayDoi",
    "TD_Symbol",
    "organ_name",
    "icb_name3",
    "icb_name4",
    "icb_code1",
    "icb_code2",
    "icb_code3",
    "icb_code4",
]

FEATURE_BASE_COLUMNS = [
    "Ngay",
    "symbol",
    "group",
    "icb_name2",
    "com_type_code",
    "LS_GiaDieuChinh",
    "LS_KhoiLuongKhopLenh",
    "LS_GiaTriKhopLenh",
]


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#B0BEC5"
    plt.rcParams["grid.color"] = "#CFD8DC"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["legend.frameon"] = True


def _to_datetime(value: Any) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    return pd.to_datetime(value)


def clean_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "Ngay" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Dữ liệu đầu vào phải có ít nhất 2 cột: 'Ngay' và 'symbol'.")

    if pd.api.types.is_datetime64_any_dtype(df["Ngay"]):
        df["Ngay"] = pd.to_datetime(df["Ngay"], errors="coerce")
    else:
        df["Ngay"] = pd.to_datetime(df["Ngay"], format="%d/%m/%Y", errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.dropna(subset=["LS_GiaDieuChinh", "Ngay"])
    df = df.drop_duplicates(subset=["Ngay", "symbol"], keep="last")
    df = df.sort_values(["symbol", "Ngay"]).reset_index(drop=True)
    return df


def load_and_clean_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    return clean_raw_dataframe(df)


def combine_history_and_recent(
    history_df: Optional[pd.DataFrame],
    recent_df: pd.DataFrame,
) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return recent_df.sort_values(["symbol", "Ngay"]).reset_index(drop=True)

    combined = pd.concat([history_df, recent_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Ngay", "symbol"], keep="last")
    combined = combined.sort_values(["symbol", "Ngay"]).reset_index(drop=True)
    return combined


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("symbol")
    adj = "LS_GiaDieuChinh"
    vol = "LS_KhoiLuongKhopLenh"

    for n in [1, 3, 5, 10]:
        df[f"return_{n}d"] = g[adj].pct_change(n)

    df["_ret1"] = g[adj].pct_change(1)
    for n in [5, 10, 20]:
        df[f"vol_{n}d"] = df.groupby("symbol")["_ret1"].transform(
            lambda x: x.rolling(n, min_periods=n // 2).std()
        )
    df = df.drop(columns="_ret1")

    for n in [5, 10, 20]:
        vol_ma = g[vol].transform(lambda x: x.rolling(n, min_periods=n // 2).mean())
        df[f"vol_ma{n}"] = vol_ma
        df[f"vol_ratio{n}"] = df[vol] / (vol_ma + 1.0)

    for n in [5, 10, 20, 50]:
        ma = g[adj].transform(lambda x: x.rolling(n, min_periods=n // 2).mean())
        df[f"ma{n}"] = ma
        df[f"close_vs_ma{n}"] = df[adj] / (ma + 1e-9) - 1.0

    open_col = df["LS_GiaMoCua"]
    high_col = df["LS_GiaCaoNhat"]
    low_col = df["LS_GiaThapNhat"]
    close_col = df["LS_GiaDongCua"]
    df["body_pct"] = (close_col - open_col) / (open_col + 1e-9)
    df["shadow_pct"] = (high_col - low_col) / (open_col + 1e-9)
    df["upper_shadow"] = (high_col - close_col.where(close_col > open_col, open_col)) / (
        open_col + 1e-9
    )
    df["lower_shadow"] = (open_col.where(close_col > open_col, close_col) - low_col) / (
        open_col + 1e-9
    )

    df["money_flow"] = df["LS_GiaTriKhopLenh"]
    df["money_flow_ma5"] = df.groupby("symbol")["money_flow"].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    df["money_flow_ratio"] = df["money_flow"] / (df["money_flow_ma5"] + 1.0)

    order_cols = {
        "DL_KLDatMua",
        "DL_KLDatBan",
        "DL_ChenhLechKL",
        "DL_KLTB1LenhMua",
        "DL_KLTB1LenhBan",
    }
    if order_cols.issubset(df.columns):
        total_order = df["DL_KLDatMua"] + df["DL_KLDatBan"] + 1.0
        df["order_imbalance"] = df["DL_ChenhLechKL"] / total_order
        df["buy_sell_ratio"] = df["DL_KLDatMua"] / (df["DL_KLDatBan"] + 1.0)
        df["avg_buy_order"] = df["DL_KLTB1LenhMua"].fillna(0)
        df["avg_sell_order"] = df["DL_KLTB1LenhBan"].fillna(0)

    if "KN_GTDGRong" in df.columns:
        df["foreign_net_value"] = df["KN_GTDGRong"].fillna(0)
        df["foreign_net_vol"] = (
            df["KN_KLGDRong"].fillna(0) if "KN_KLGDRong" in df.columns else 0.0
        )
        df["foreign_net_ma5"] = df.groupby("symbol")["foreign_net_value"].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        if "KN_KLMua" in df.columns:
            df["foreign_buy_ratio"] = df["KN_KLMua"] / (df[vol] + 1.0)
        if {"KN_RoomConLai", "KN_DangSoHuu"}.issubset(df.columns):
            df["foreign_room_pct"] = df["KN_RoomConLai"] / (
                df["KN_DangSoHuu"] + df["KN_RoomConLai"] + 1.0
            )

    if {"TD_KLcpMua", "TD_KlcpBan", "TD_GtMua", "TD_GtBan"}.issubset(df.columns):
        df["block_net_vol"] = df["TD_KLcpMua"].fillna(0) - df["TD_KlcpBan"].fillna(0)
        df["block_net_value"] = df["TD_GtMua"].fillna(0) - df["TD_GtBan"].fillna(0)
        df["block_vs_vol"] = df["TD_KLcpMua"].fillna(0) / (df[vol] + 1.0)

    df["rsi_14"] = g[adj].transform(lambda x: compute_rsi(x, 14))

    ema12 = g[adj].transform(lambda x: x.ewm(span=12, min_periods=8).mean())
    ema26 = g[adj].transform(lambda x: x.ewm(span=26, min_periods=18).mean())
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("symbol")["macd"].transform(
        lambda x: x.ewm(span=9, min_periods=6).mean()
    )
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    encoder = LabelEncoder()
    if "icb_name2" in df.columns:
        df["sector_code"] = encoder.fit_transform(df["icb_name2"].fillna("Unknown"))
    if "com_type_code" in df.columns:
        df["company_type"] = encoder.fit_transform(df["com_type_code"].fillna("CT"))
    df["day_of_week"] = df["Ngay"].dt.dayofweek
    df["month"] = df["Ngay"].dt.month

    rank_cols = [
        "return_5d",
        "vol_ratio10",
        "foreign_net_value",
        "money_flow_ratio",
        "rsi_14",
        "order_imbalance",
    ]
    for col in rank_cols:
        if col in df.columns:
            df[f"rank_{col}"] = df.groupby("Ngay")[col].rank(pct=True, na_option="keep")

    if "ma20" in df.columns:
        df["breakout_ma20"] = (df["LS_GiaDieuChinh"] > df["ma20"]).astype(float)
    if "vol_ratio10" in df.columns:
        df["volume_spike"] = (df["vol_ratio10"] > 2.0).astype(float)
    if "rsi_14" in df.columns:
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(float)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
    if "order_imbalance" in df.columns:
        df["order_imbalance_extreme"] = (df["order_imbalance"].abs() > 0.30).astype(
            float
        )
    if "foreign_net_value" in df.columns:
        fn_ma20 = df.groupby("symbol")["foreign_net_value"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        df["foreign_buy_spike"] = (df["foreign_net_value"] > 2.0 * fn_ma20).astype(
            float
        )

    return df


def _cycle_edges_for_k(
    k: int,
    ref_horizon: int = 3,
    base_edges: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if base_edges is None:
        base_edges = CYCLE_BASE_EDGES
    scale = np.sqrt(float(k) / max(float(ref_horizon), 1.0))
    scaled = np.asarray(base_edges, dtype=np.float32) * scale
    return np.concatenate(([-np.inf], scaled, [np.inf]))


def add_supplemental_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("symbol")

    delta_cols = [
        "vol_ratio10",
        "foreign_net_value",
        "order_imbalance",
        "rsi_14",
        "macd_hist",
        "money_flow_ratio",
    ]
    for col in delta_cols:
        if col in df.columns:
            df[f"d1_{col}"] = g[col].diff(1)
            df[f"d3_{col}"] = g[col].diff(3)
            df[f"lag1_{col}"] = g[col].shift(1)
            df[f"lag2_{col}"] = g[col].shift(2)
            df[f"lag3_{col}"] = g[col].shift(3)

    if "return_1d" in df.columns:
        market_ret_1d = df.groupby("Ngay")["return_1d"].transform("mean")
        df["alpha_1d"] = df["return_1d"] - market_ret_1d
    if "return_5d" in df.columns:
        market_ret_5d = df.groupby("Ngay")["return_5d"].transform("mean")
        df["alpha_5d"] = df["return_5d"] - market_ret_5d
        df["stock_return_minus_market_return"] = df["alpha_5d"]
    if "return_10d" in df.columns:
        market_ret_10d = df.groupby("Ngay")["return_10d"].transform("mean")
        df["alpha_10d"] = df["return_10d"] - market_ret_10d
        df["relative_strength_vs_market"] = df["alpha_10d"]

    if "icb_name2" in df.columns and "return_1d" in df.columns:
        sector_ret_1d = df.groupby(["Ngay", "icb_name2"])["return_1d"].transform("mean")
        df["stock_return_minus_sector_return"] = df["return_1d"] - sector_ret_1d
    if "icb_name2" in df.columns and "vol_ratio10" in df.columns:
        sector_vol_ratio = df.groupby(["Ngay", "icb_name2"])["vol_ratio10"].transform(
            "mean"
        )
        df["stock_volume_ratio_vs_sector"] = df["vol_ratio10"] / (
            sector_vol_ratio + 1e-9
        )
    if "icb_name2" in df.columns and "money_flow_ratio" in df.columns:
        sector_money_flow = df.groupby(["Ngay", "icb_name2"])["money_flow_ratio"].transform(
            "mean"
        )
        df["stock_money_flow_vs_sector"] = df["money_flow_ratio"] - sector_money_flow

    if "return_5d" in df.columns and "vol_ratio10" in df.columns:
        df["momentum_x_volume"] = df["return_5d"] * df["vol_ratio10"]
    if "return_5d" in df.columns and "foreign_net_value" in df.columns:
        df["momentum_x_foreign_flow"] = df["return_5d"] * df["foreign_net_value"]
    if "order_imbalance" in df.columns and "breakout_ma20" in df.columns:
        df["orderflow_x_breakout"] = df["order_imbalance"] * df["breakout_ma20"]
    if "money_flow_ratio" in df.columns and "vol_10d" in df.columns:
        df["moneyflow_x_volatility"] = df["money_flow_ratio"] * df["vol_10d"]

    if "return_1d" in df.columns:
        market_daily = df.groupby("Ngay")["return_1d"].mean().sort_index()
        market_index = (1.0 + market_daily.fillna(0.0)).cumprod()
        market_ma20 = market_index.rolling(20, min_periods=10).mean()
        market_flag = (market_index > market_ma20).astype(float)
        df["market_uptrend_flag"] = df["Ngay"].map(market_flag)

    return df


def add_cycle_regime_features(
    df: pd.DataFrame,
    k_values: Sequence[int] = CYCLE_K_VALUES,
    n_values: Sequence[int] = CYCLE_N_VALUES,
    price_col: str = "LS_GiaDieuChinh",
) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("symbol")
    state_names = [
        "dn_big",
        "dn_mid",
        "dn_small",
        "flat",
        "up_small",
        "up_mid",
        "up_big",
    ]

    for k in k_values:
        cycle_ret_col = f"cycle_ret_k{k}"
        df[cycle_ret_col] = g[price_col].pct_change(k)
        df[f"cycle_abs_ret_k{k}"] = df[cycle_ret_col].abs()
        df[f"cycle_dir_k{k}"] = np.sign(df[cycle_ret_col]).astype(np.float32)

        cycle_group = df.groupby("symbol")[cycle_ret_col]
        for n_cycles in n_values:
            hist = np.column_stack(
                [cycle_group.shift(i * k).to_numpy(dtype=np.float32) for i in range(n_cycles)]
            )
            valid = np.isfinite(hist)
            count = valid.sum(axis=1).astype(np.float32)
            safe = np.clip(count, 1.0, None)
            hist_zero = np.where(valid, hist, 0.0).astype(np.float32)

            mean_ret = hist_zero.sum(axis=1) / safe
            sq_mean = (hist_zero * hist_zero).sum(axis=1) / safe
            std_ret = np.sqrt(np.maximum(sq_mean - mean_ret * mean_ret, 0.0))
            up_frac = (valid & (hist > 0)).sum(axis=1) / safe
            down_frac = (valid & (hist < 0)).sum(axis=1) / safe
            non_up_frac = (valid & (hist <= 0)).sum(axis=1) / safe

            weights = np.arange(n_cycles, 0, -1, dtype=np.float32).reshape(1, -1)
            weight_valid = (valid * weights).sum(axis=1)
            weighted_score = (np.where(valid, np.sign(hist), 0.0) * weights).sum(axis=1)
            weighted_score = np.divide(weighted_score, np.clip(weight_valid, 1.0, None))

            mean_ret[count == 0] = np.nan
            std_ret[count == 0] = np.nan
            up_frac[count == 0] = np.nan
            down_frac[count == 0] = np.nan
            non_up_frac[count == 0] = np.nan
            weighted_score[count == 0] = np.nan

            df[f"cyc_mean_ret_k{k}_n{n_cycles}"] = mean_ret
            df[f"cyc_std_ret_k{k}_n{n_cycles}"] = std_ret
            df[f"cyc_up_frac_k{k}_n{n_cycles}"] = up_frac
            df[f"cyc_down_frac_k{k}_n{n_cycles}"] = down_frac
            df[f"cyc_non_up_frac_k{k}_n{n_cycles}"] = non_up_frac
            df[f"cyc_pos_minus_neg_k{k}_n{n_cycles}"] = up_frac - down_frac
            df[f"cyc_weighted_score_k{k}_n{n_cycles}"] = weighted_score

            bounds = _cycle_edges_for_k(k)
            for idx, state_name in enumerate(state_names):
                lo = bounds[idx]
                hi = bounds[idx + 1]
                if np.isneginf(lo):
                    mask = valid & (hist <= hi)
                elif np.isposinf(hi):
                    mask = valid & (hist > lo)
                else:
                    mask = valid & (hist > lo) & (hist <= hi)
                frac = mask.sum(axis=1) / safe
                frac[count == 0] = np.nan
                df[f"cyc_{state_name}_frac_k{k}_n{n_cycles}"] = frac

    return df


def build_feature_table(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = compute_base_features(clean_df)
    df = add_supplemental_features(df)
    df = add_cycle_regime_features(df)
    return df


def add_realized_outcomes(
    df: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    price_col: str = "LS_GiaDieuChinh",
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("symbol")

    for horizon in horizons:
        next_price = g[price_col].shift(-horizon)
        future_return = (next_price - out[price_col]) / (out[price_col] + 1e-9)
        out[f"return_t{horizon}"] = future_return

        median_ret = out.groupby("Ngay")[f"return_t{horizon}"].transform("median")
        label = pd.Series(np.nan, index=out.index, dtype="float64")
        valid = future_return.notna() & median_ret.notna()
        label.loc[valid] = (future_return.loc[valid] > median_ret.loc[valid]).astype(float)
        out[f"label_t{horizon}"] = label.astype("Float64")

    return out


def load_model_bundle(
    models_dir: str | Path,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> Dict[int, Dict[str, Any]]:
    models_dir = Path(models_dir)
    bundle: Dict[int, Dict[str, Any]] = {}
    for horizon in horizons:
        model_path = models_dir / f"lgbm_t{horizon}.txt"
        meta_path = models_dir / f"meta_t{horizon}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Không tìm thấy meta: {meta_path}")

        model = lgb.Booster(model_file=str(model_path))
        with open(meta_path, "rb") as file:
            meta = pickle.load(file)

        bundle[horizon] = {"model": model, "meta": meta}
    return bundle


def _ensure_selected_features(
    frame: pd.DataFrame,
    selected_features: Sequence[str],
) -> pd.DataFrame:
    score_frame = frame.copy()
    missing = [col for col in selected_features if col not in score_frame.columns]
    for col in missing:
        score_frame[col] = np.nan
    if missing:
        warnings.warn(
            "Một số feature trong model không xuất hiện ở dữ liệu áp dụng. "
            f"Chúng sẽ được thêm với giá trị NaN: {missing[:10]}"
            + (" ..." if len(missing) > 10 else ""),
            stacklevel=2,
        )
    score_frame = score_frame.loc[:, list(selected_features)].apply(
        pd.to_numeric, errors="coerce"
    )
    return score_frame


def _apply_daily_rank_and_signal(
    df: pd.DataFrame,
    score_col: str,
    prefix: str,
    buy_threshold: float = BUY_THRESHOLD,
    avoid_threshold: float = AVOID_THRESHOLD,
) -> pd.DataFrame:
    out = df.copy()
    out[f"rank_pct_{prefix}"] = out.groupby("Ngay")[score_col].rank(
        ascending=True,
        pct=True,
        method="first",
    )
    out[f"rank_{prefix}"] = out.groupby("Ngay")[score_col].rank(
        ascending=False,
        method="first",
    )
    out[f"signal_{prefix}"] = np.select(
        [
            out[f"rank_pct_{prefix}"] >= buy_threshold,
            out[f"rank_pct_{prefix}"] <= avoid_threshold,
        ],
        ["BUY", "AVOID"],
        default="WATCH",
    )
    return out


def _make_latest_table(
    df: pd.DataFrame,
    score_col: str,
    signal_col: str,
    rank_col: str,
    top_n: int = TOP_N_LATEST,
) -> pd.DataFrame:
    use_cols = [
        "Ngay",
        "symbol",
        "group",
        "icb_name2",
        "com_type_code",
        "LS_GiaDieuChinh",
        score_col,
        signal_col,
        rank_col,
        "agreement_buy_count",
        "avg_prob_up",
    ]
    use_cols = [col for col in use_cols if col in df.columns]
    latest = df.sort_values(score_col, ascending=False).head(top_n).copy()
    return latest[use_cols].reset_index(drop=True)


def _calculate_daily_topk_curve(
    pred_df: pd.DataFrame,
    horizon: int,
    top_k: int = 5,
) -> pd.DataFrame:
    prob_col = f"prob_up_t{horizon}"
    ret_col = f"return_t{horizon}"
    valid = pred_df.dropna(subset=[prob_col, ret_col]).copy()
    if valid.empty:
        return pd.DataFrame()

    top = (
        valid.sort_values(["Ngay", prob_col], ascending=[True, False])
        .groupby("Ngay")
        .head(top_k)
        .groupby("Ngay")
        .agg(strategy_return=(ret_col, "mean"), hit_rate=(f"label_t{horizon}", "mean"))
        .reset_index()
    )
    top["cum_return"] = (1.0 + top["strategy_return"].fillna(0.0)).cumprod()
    top["horizon"] = horizon
    return top


def _build_decile_table(pred_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    prob_col = f"prob_up_t{horizon}"
    ret_col = f"return_t{horizon}"
    valid = pred_df.dropna(subset=[prob_col, ret_col]).copy()
    if valid.empty:
        return pd.DataFrame()

    valid["decile"] = valid.groupby("Ngay")[prob_col].transform(
        lambda x: pd.qcut(
            x.rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        )
        + 1
    )
    decile_df = (
        valid.groupby("decile")
        .agg(
            avg_future_return=(ret_col, "mean"),
            hit_rate=(f"label_t{horizon}", "mean"),
            observations=(ret_col, "count"),
        )
        .reset_index()
    )
    decile_df["horizon"] = horizon
    return decile_df


def _feature_coverage_for_selected(
    score_df: pd.DataFrame,
    bundle: Dict[int, Dict[str, Any]],
    top_n: int = 20,
) -> pd.DataFrame:
    rows = []
    for horizon, info in bundle.items():
        selected = list(info["meta"]["features"])
        available = [col for col in selected if col in score_df.columns]
        if not available:
            continue

        variances = score_df[available].apply(pd.to_numeric, errors="coerce").var()
        top_features = variances.sort_values(ascending=False).head(top_n).index.tolist()
        coverage = score_df[top_features].notna().mean()
        for feature_name, cov in coverage.items():
            rows.append(
                {
                    "horizon": horizon,
                    "feature": feature_name,
                    "coverage": float(cov),
                }
            )
    return pd.DataFrame(rows)


def predict_from_clean_data(
    clean_df: pd.DataFrame,
    models_dir: str | Path,
    score_start: Optional[str | pd.Timestamp] = None,
    score_end: Optional[str | pd.Timestamp] = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    top_n_latest: int = TOP_N_LATEST,
) -> Dict[str, Any]:
    bundle = load_model_bundle(models_dir=models_dir, horizons=horizons)
    feature_df = build_feature_table(clean_df)
    feature_df = add_realized_outcomes(feature_df, horizons=horizons)

    if score_start is None:
        score_start = feature_df["Ngay"].min()
    if score_end is None:
        score_end = feature_df["Ngay"].max()

    score_start = _to_datetime(score_start)
    score_end = _to_datetime(score_end)

    score_mask = (feature_df["Ngay"] >= score_start) & (feature_df["Ngay"] <= score_end)
    score_df = feature_df.loc[score_mask].copy()
    if score_df.empty:
        raise ValueError("Khoảng thời gian cần dự đoán không có dữ liệu.")

    for horizon, info in bundle.items():
        selected_features = list(info["meta"]["features"])
        score_matrix = _ensure_selected_features(score_df, selected_features)
        prob_col = f"prob_up_t{horizon}"
        score_df[prob_col] = info["model"].predict(score_matrix)
        score_df = _apply_daily_rank_and_signal(
            score_df,
            score_col=prob_col,
            prefix=f"t{horizon}",
        )

    prob_cols = [f"prob_up_t{h}" for h in horizons if f"prob_up_t{h}" in score_df.columns]
    rank_pct_cols = [f"rank_pct_t{h}" for h in horizons if f"rank_pct_t{h}" in score_df.columns]

    score_df["avg_prob_up"] = score_df[prob_cols].mean(axis=1)
    score_df["ensemble_score"] = score_df[rank_pct_cols].mean(axis=1)
    score_df["agreement_buy_count"] = (
        sum((score_df[f"signal_t{h}"] == "BUY").astype(int) for h in horizons)
        if horizons
        else 0
    )
    score_df = _apply_daily_rank_and_signal(
        score_df,
        score_col="ensemble_score",
        prefix="ensemble",
    )

    keep_cols = FEATURE_BASE_COLUMNS + [
        col
        for col in score_df.columns
        if col.startswith(("prob_up_t", "rank_pct_t", "rank_t", "signal_t", "return_t", "label_t"))
    ] + [
        "avg_prob_up",
        "ensemble_score",
        "rank_pct_ensemble",
        "rank_ensemble",
        "signal_ensemble",
        "agreement_buy_count",
    ]
    keep_cols = [col for col in keep_cols if col in score_df.columns]
    prediction_df = score_df[keep_cols].copy()

    latest_date = prediction_df["Ngay"].max()
    latest_df = prediction_df[prediction_df["Ngay"] == latest_date].copy()

    latest_tables: Dict[str, pd.DataFrame] = {}
    latest_tables["ensemble"] = _make_latest_table(
        latest_df,
        score_col="ensemble_score",
        signal_col="signal_ensemble",
        rank_col="rank_ensemble",
        top_n=top_n_latest,
    )
    for horizon in horizons:
        latest_tables[f"T+{horizon}"] = _make_latest_table(
            latest_df,
            score_col=f"prob_up_t{horizon}",
            signal_col=f"signal_t{horizon}",
            rank_col=f"rank_t{horizon}",
            top_n=top_n_latest,
        )

    evaluation_summary = []
    decile_tables: Dict[int, pd.DataFrame] = {}
    topk_curves: Dict[int, pd.DataFrame] = {}

    for horizon in horizons:
        decile_df = _build_decile_table(prediction_df, horizon)
        decile_tables[horizon] = decile_df

        curve_df = _calculate_daily_topk_curve(prediction_df, horizon, top_k=5)
        topk_curves[horizon] = curve_df

        prob_col = f"prob_up_t{horizon}"
        ret_col = f"return_t{horizon}"
        label_col = f"label_t{horizon}"
        valid = prediction_df.dropna(subset=[prob_col, ret_col]).copy()

        if valid.empty:
            evaluation_summary.append(
                {
                    "horizon": horizon,
                    "valid_rows": 0,
                    "avg_top5_return": np.nan,
                    "avg_top5_hit_rate": np.nan,
                    "avg_bottom5_return": np.nan,
                    "top_bottom_spread": np.nan,
                }
            )
            continue

        top = (
            valid.sort_values(["Ngay", prob_col], ascending=[True, False])
            .groupby("Ngay")
            .head(5)
        )
        bottom = (
            valid.sort_values(["Ngay", prob_col], ascending=[True, True])
            .groupby("Ngay")
            .head(5)
        )
        evaluation_summary.append(
            {
                "horizon": horizon,
                "valid_rows": int(len(valid)),
                "avg_top5_return": float(top[ret_col].mean()),
                "avg_top5_hit_rate": float(top[label_col].mean()) if label_col in top.columns else np.nan,
                "avg_bottom5_return": float(bottom[ret_col].mean()),
                "top_bottom_spread": float(top[ret_col].mean() - bottom[ret_col].mean()),
            }
        )

    feature_coverage = _feature_coverage_for_selected(score_df, bundle=bundle)
    trading_days_before_window = clean_df.loc[clean_df["Ngay"] < score_start, "Ngay"].nunique()

    summary = {
        "window_start": score_start.strftime("%Y-%m-%d"),
        "window_end": score_end.strftime("%Y-%m-%d"),
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "rows_scored": int(len(prediction_df)),
        "symbols_scored": int(prediction_df["symbol"].nunique()),
        "trading_days_scored": int(prediction_df["Ngay"].nunique()),
        "history_trading_days_before_window": int(trading_days_before_window),
        "lookback_warning": bool(trading_days_before_window < MIN_LOOKBACK_TRADING_DAYS),
    }

    result = {
        "summary": summary,
        "bundle": bundle,
        "feature_df": feature_df,
        "prediction_df": prediction_df.sort_values(["Ngay", "symbol"]).reset_index(drop=True),
        "latest_df": latest_df.sort_values("ensemble_score", ascending=False).reset_index(drop=True),
        "latest_tables": latest_tables,
        "decile_tables": decile_tables,
        "topk_curves": topk_curves,
        "evaluation_summary": pd.DataFrame(evaluation_summary),
        "feature_coverage": feature_coverage,
    }
    result["api_payload"] = build_api_payload(result)
    return result


def predict_from_csv(
    history_csv_path: str | Path,
    recent_csv_path: str | Path,
    models_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> Dict[str, Any]:
    history_df = load_and_clean_csv(history_csv_path)
    recent_df = load_and_clean_csv(recent_csv_path)
    combined_df = combine_history_and_recent(history_df, recent_df)

    score_start = recent_df["Ngay"].min()
    score_end = recent_df["Ngay"].max()

    result = predict_from_clean_data(
        clean_df=combined_df,
        models_dir=models_dir,
        score_start=score_start,
        score_end=score_end,
        horizons=horizons,
    )
    result["history_df"] = history_df
    result["recent_df"] = recent_df
    result["combined_df"] = combined_df

    if output_dir is not None:
        result["exported_files"] = export_prediction_outputs(result, output_dir=output_dir)
    else:
        result["exported_files"] = {}

    return result


def build_api_payload(
    result: Dict[str, Any],
    top_n: int = TOP_N_LATEST,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "window_start": result["summary"]["window_start"],
        "window_end": result["summary"]["window_end"],
        "latest_date": result["summary"]["latest_date"],
        "symbols_scored": result["summary"]["symbols_scored"],
        "trading_days_scored": result["summary"]["trading_days_scored"],
        "lookback_warning": result["summary"]["lookback_warning"],
        "recommendations": {},
    }

    for view_name, table_df in result["latest_tables"].items():
        payload["recommendations"][view_name] = (
            table_df.head(top_n)
            .assign(Ngay=lambda x: x["Ngay"].dt.strftime("%Y-%m-%d"))
            .replace({np.nan: None})
            .to_dict(orient="records")
        )
    return payload


def export_prediction_outputs(
    result: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = output_dir / "predictions_recent_month.csv"
    latest_path = output_dir / "latest_day_top_picks.csv"
    eval_path = output_dir / "evaluation_summary.csv"
    coverage_path = output_dir / "feature_coverage.csv"
    payload_path = output_dir / "latest_day_api_payload.json"

    prediction_df = result["prediction_df"].copy()
    prediction_df["Ngay"] = prediction_df["Ngay"].dt.strftime("%Y-%m-%d")
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    latest_frames = []
    for view_name, df_view in result["latest_tables"].items():
        temp = df_view.copy()
        temp.insert(0, "view", view_name)
        if "Ngay" in temp.columns:
            temp["Ngay"] = temp["Ngay"].dt.strftime("%Y-%m-%d")
        latest_frames.append(temp)
    pd.concat(latest_frames, ignore_index=True).to_csv(
        latest_path,
        index=False,
        encoding="utf-8-sig",
    )

    result["evaluation_summary"].to_csv(eval_path, index=False, encoding="utf-8-sig")
    result["feature_coverage"].to_csv(coverage_path, index=False, encoding="utf-8-sig")

    with open(payload_path, "w", encoding="utf-8") as file:
        json.dump(result["api_payload"], file, ensure_ascii=False, indent=2)

    return {
        "predictions_recent_month": str(prediction_path),
        "latest_day_top_picks": str(latest_path),
        "evaluation_summary": str(eval_path),
        "feature_coverage": str(coverage_path),
        "latest_day_api_payload": str(payload_path),
    }


def plot_prediction_overview(result: Dict[str, Any]) -> plt.Figure:
    configure_plotting()
    pred_df = result["prediction_df"].copy()
    horizons = sorted(result["bundle"].keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle("Tổng quan dự đoán trong 1 tháng gần nhất", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    for horizon in horizons:
        prob_col = f"prob_up_t{horizon}"
        daily = pred_df.groupby("Ngay")[prob_col].count()
        ax1.plot(daily.index, daily.values, marker="o", linewidth=2, label=f"T+{horizon}")
    ax1.set_title("Số lượng mã được chấm điểm theo ngày")
    ax1.set_xlabel("Ngày")
    ax1.set_ylabel("Số mã")
    ax1.legend(title="Horizon")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = axes[0, 1]
    prob_long = pred_df.melt(
        id_vars=["Ngay", "symbol"],
        value_vars=[f"prob_up_t{h}" for h in horizons],
        var_name="horizon",
        value_name="probability",
    )
    prob_long["horizon"] = prob_long["horizon"].str.replace("prob_up_t", "T+", regex=False)
    sns.boxplot(data=prob_long, x="horizon", y="probability", palette="Blues", ax=ax2)
    ax2.set_title("Phân bố xác suất dự đoán theo từng horizon")
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("Xác suất vượt trung vị")

    ax3 = axes[1, 0]
    for horizon in horizons:
        prob_col = f"prob_up_t{horizon}"
        daily_avg = pred_df.groupby("Ngay")[prob_col].mean()
        ax3.plot(daily_avg.index, daily_avg.values, linewidth=2, label=f"T+{horizon}")
    ax3.axhline(0.5, color="#EF6C00", linestyle="--", linewidth=1.5, label="Mốc trung tính 0.5")
    ax3.set_title("Xác suất trung bình theo ngày")
    ax3.set_xlabel("Ngày")
    ax3.set_ylabel("Xác suất trung bình")
    ax3.legend(title="Horizon")
    ax3.tick_params(axis="x", rotation=45)

    ax4 = axes[1, 1]
    for horizon in horizons:
        prob_col = f"prob_up_t{horizon}"
        daily_top1 = pred_df.groupby("Ngay")[prob_col].max()
        ax4.plot(daily_top1.index, daily_top1.values, marker="o", linewidth=2, label=f"T+{horizon}")
    ax4.set_title("Độ tự tin của mã đứng đầu mỗi ngày")
    ax4.set_xlabel("Ngày")
    ax4.set_ylabel("Xác suất cao nhất trong ngày")
    ax4.legend(title="Horizon")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def plot_latest_day_dashboard(result: Dict[str, Any], top_n: int = TOP_N_LATEST) -> plt.Figure:
    configure_plotting()
    latest_df = result["latest_df"].copy()
    latest_date = result["summary"]["latest_date"]
    horizons = sorted(result["bundle"].keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(
        f"Bảng điều khiển dự đoán cho ngày mới nhất: {latest_date}",
        fontsize=16,
        fontweight="bold",
    )

    ax1 = axes[0, 0]
    top_ensemble = result["latest_tables"]["ensemble"].head(top_n).iloc[::-1]
    ax1.barh(top_ensemble["symbol"], top_ensemble["ensemble_score"], color="#1565C0", alpha=0.9)
    ax1.set_title("Top mã theo điểm tổng hợp ensemble")
    ax1.set_xlabel("Điểm ensemble")
    ax1.set_ylabel("Mã cổ phiếu")

    ax2 = axes[0, 1]
    union_symbols = result["latest_tables"]["ensemble"]["symbol"].head(top_n).tolist()
    heat_df = latest_df[latest_df["symbol"].isin(union_symbols)].copy()
    heat_matrix = heat_df.set_index("symbol")[[f"prob_up_t{h}" for h in horizons]].rename(
        columns={f"prob_up_t{h}": f"T+{h}" for h in horizons}
    )
    if "T+3" in heat_matrix.columns:
        heat_matrix = heat_matrix.sort_values("T+3", ascending=False)
    sns.heatmap(
        heat_matrix,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Xác suất"},
        ax=ax2,
    )
    ax2.set_title("Mức đồng thuận xác suất giữa các horizon")
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("Mã cổ phiếu")

    ax3 = axes[1, 0]
    sector_count = (
        result["latest_tables"]["ensemble"]["icb_name2"]
        .fillna("Không rõ")
        .value_counts()
        .head(10)
        .sort_values()
    )
    ax3.barh(sector_count.index, sector_count.values, color="#2E7D32", alpha=0.9)
    ax3.set_title("Cơ cấu ngành của top mã ngày mới nhất")
    ax3.set_xlabel("Số lượng mã")
    ax3.set_ylabel("Ngành")

    ax4 = axes[1, 1]
    agreement = latest_df["agreement_buy_count"].value_counts().sort_index()
    ax4.bar(agreement.index.astype(str), agreement.values, color="#EF6C00", alpha=0.9)
    ax4.set_title("Mức đồng thuận BUY giữa các mô hình")
    ax4.set_xlabel("Số horizon cùng cho tín hiệu BUY")
    ax4.set_ylabel("Số lượng mã")

    plt.tight_layout()
    return fig


def plot_realized_quality(result: Dict[str, Any]) -> plt.Figure:
    configure_plotting()
    eval_df = result["evaluation_summary"].copy()
    horizons = sorted(result["bundle"].keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(
        "Đánh giá chất lượng dự đoán trên phần dữ liệu đã có kết quả thực tế",
        fontsize=16,
        fontweight="bold",
    )

    ax1 = axes[0, 0]
    has_decile_data = False
    for horizon in horizons:
        decile_df = result["decile_tables"].get(horizon, pd.DataFrame())
        if decile_df.empty:
            continue
        has_decile_data = True
        ax1.plot(
            decile_df["decile"],
            decile_df["avg_future_return"] * 100,
            marker="o",
            linewidth=2,
            label=f"T+{horizon}",
        )
    ax1.set_title("Lợi nhuận tương lai trung bình theo decile xác suất")
    ax1.set_xlabel("Decile xác suất")
    ax1.set_ylabel("Lợi nhuận tương lai trung bình (%)")
    if has_decile_data:
        ax1.legend(title="Horizon")
    else:
        ax1.text(0.5, 0.5, "Chưa đủ dữ liệu realized", ha="center", va="center")

    ax2 = axes[0, 1]
    has_curve_data = False
    for horizon in horizons:
        curve_df = result["topk_curves"].get(horizon, pd.DataFrame())
        if curve_df.empty:
            continue
        has_curve_data = True
        ax2.plot(curve_df["Ngay"], curve_df["cum_return"], marker="o", linewidth=2, label=f"T+{horizon}")
    ax2.set_title("Lợi nhuận lũy kế nếu mỗi ngày giữ top-5 mã")
    ax2.set_xlabel("Ngày")
    ax2.set_ylabel("Giá trị tài sản tích lũy")
    if has_curve_data:
        ax2.legend(title="Horizon")
        ax2.tick_params(axis="x", rotation=45)
    else:
        ax2.text(0.5, 0.5, "Chưa đủ dữ liệu realized", ha="center", va="center")

    ax3 = axes[1, 0]
    eval_plot = eval_df.dropna(subset=["avg_top5_return", "avg_bottom5_return"]).copy()
    if not eval_plot.empty:
        x = np.arange(len(eval_plot))
        width = 0.35
        ax3.bar(x - width / 2, eval_plot["avg_top5_return"] * 100, width=width, label="Top 5", color="#1565C0")
        ax3.bar(x + width / 2, eval_plot["avg_bottom5_return"] * 100, width=width, label="Bottom 5", color="#C62828")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"T+{h}" for h in eval_plot["horizon"]])
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Chưa đủ dữ liệu realized", ha="center", va="center")
    ax3.set_title("So sánh lợi nhuận top 5 và bottom 5")
    ax3.set_xlabel("Horizon")
    ax3.set_ylabel("Lợi nhuận trung bình (%)")

    ax4 = axes[1, 1]
    if not eval_plot.empty:
        ax4.bar([f"T+{h}" for h in eval_plot["horizon"]], eval_plot["avg_top5_hit_rate"] * 100, color="#2E7D32", alpha=0.9)
        ax4.axhline(50, color="#EF6C00", linestyle="--", linewidth=1.5, label="Mốc ngẫu nhiên 50%")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "Chưa đủ dữ liệu realized", ha="center", va="center")
    ax4.set_title("Hit rate của top 5 mã được chọn")
    ax4.set_xlabel("Horizon")
    ax4.set_ylabel("Tỷ lệ vượt trung vị (%)")

    plt.tight_layout()
    return fig


def make_summary_table(result: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Cửa sổ dự đoán", f"{result['summary']['window_start']} -> {result['summary']['window_end']}"],
            ["Ngày mới nhất trong cửa sổ", result["summary"]["latest_date"]],
            ["Tổng số dòng được chấm điểm", f"{result['summary']['rows_scored']:,}"],
            ["Số mã cổ phiếu", f"{result['summary']['symbols_scored']:,}"],
            ["Số phiên giao dịch", f"{result['summary']['trading_days_scored']:,}"],
            ["Số phiên lịch sử trước cửa sổ", f"{result['summary']['history_trading_days_before_window']:,}"],
            ["Cảnh báo lookback", "Có" if result["summary"]["lookback_warning"] else "Không"],
        ],
        columns=["Chỉ tiêu", "Giá trị"],
    )


def _resolve_score_window_from_master(
    clean_df: pd.DataFrame,
    mode: str = "recent_months",
    recent_months: int = DEFAULT_SYMBOL_LOOKBACK_MONTHS,
    recent_days: int = 22,
    score_start: Optional[str | pd.Timestamp] = None,
    score_end: Optional[str | pd.Timestamp] = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_days = pd.Index(sorted(pd.to_datetime(clean_df["Ngay"].dropna().unique())))
    if len(unique_days) == 0:
        raise ValueError("Dữ liệu đầu vào không có ngày giao dịch hợp lệ.")

    mode = str(mode).strip().lower()

    if mode == "latest_day":
        latest_day = pd.Timestamp(unique_days[-1])
        return latest_day, latest_day

    if mode == "recent_months":
        recent_months = int(recent_months)
        if recent_months <= 0:
            raise ValueError("recent_months phải lớn hơn 0.")
        end_day = pd.Timestamp(unique_days[-1])
        start_day = end_day - pd.DateOffset(months=recent_months)
        eligible = unique_days[unique_days >= start_day]
        if len(eligible) == 0:
            return pd.Timestamp(unique_days[0]), end_day
        return pd.Timestamp(eligible[0]), end_day

    if mode == "recent_days":
        recent_days = int(recent_days)
        if recent_days <= 0:
            raise ValueError("recent_days phải lớn hơn 0.")
        start_idx = max(len(unique_days) - recent_days, 0)
        return pd.Timestamp(unique_days[start_idx]), pd.Timestamp(unique_days[-1])

    if mode == "date_range":
        if score_start is None or score_end is None:
            raise ValueError("Mode 'date_range' cần cả score_start và score_end.")
        return _to_datetime(score_start), _to_datetime(score_end)

    if mode == "full":
        return pd.Timestamp(unique_days[0]), pd.Timestamp(unique_days[-1])

    raise ValueError(
        "mode không hợp lệ. Chọn một trong các giá trị: "
        "'latest_day', 'recent_months', 'recent_days', 'date_range', 'full'."
    )


def predict_from_master_csv(
    master_csv_path: str | Path,
    models_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    mode: str = "recent_months",
    recent_months: int = DEFAULT_SYMBOL_LOOKBACK_MONTHS,
    recent_days: int = 22,
    score_start: Optional[str | pd.Timestamp] = None,
    score_end: Optional[str | pd.Timestamp] = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    top_n_latest: int = TOP_N_LATEST,
) -> Dict[str, Any]:
    """
    API đơn giản nhất cho user cuối:
    - Chỉ cần 1 file CSV master đã chứa lịch sử từ quá khứ đến hiện tại.
    - Hàm sẽ tự clean dữ liệu, tự tạo feature, tự predict và tự export output nếu có output_dir.

    Các mode phổ biến:
    - latest_day: chỉ dự đoán cho ngày mới nhất trong file CSV.
    - recent_months: dự đoán cho N tháng gần nhất, mặc định 24 tháng.
    - recent_days: dự đoán cho N phiên gần nhất.
    - date_range: dự đoán cho một khoảng ngày cụ thể.
    - full: dự đoán toàn bộ lịch sử có trong file.
    """
    clean_df = load_and_clean_csv(master_csv_path)
    resolved_start, resolved_end = _resolve_score_window_from_master(
        clean_df=clean_df,
        mode=mode,
        recent_months=recent_months,
        recent_days=recent_days,
        score_start=score_start,
        score_end=score_end,
    )

    result = predict_from_clean_data(
        clean_df=clean_df,
        models_dir=models_dir,
        score_start=resolved_start,
        score_end=resolved_end,
        horizons=horizons,
        top_n_latest=top_n_latest,
    )
    result["master_df"] = clean_df
    result["run_mode"] = mode

    if output_dir is not None:
        result["exported_files"] = export_prediction_outputs(result, output_dir=output_dir)
    else:
        result["exported_files"] = {}

    return result


def predict_latest_from_master_csv(
    master_csv_path: str | Path,
    models_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    top_n_latest: int = TOP_N_LATEST,
) -> Dict[str, Any]:
    return predict_from_master_csv(
        master_csv_path=master_csv_path,
        models_dir=models_dir,
        output_dir=output_dir,
        mode="latest_day",
        horizons=horizons,
        top_n_latest=top_n_latest,
    )


def predict_recent_from_master_csv(
    master_csv_path: str | Path,
    models_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    recent_months: int = DEFAULT_SYMBOL_LOOKBACK_MONTHS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    top_n_latest: int = TOP_N_LATEST,
) -> Dict[str, Any]:
    return predict_from_master_csv(
        master_csv_path=master_csv_path,
        models_dir=models_dir,
        output_dir=output_dir,
        mode="recent_months",
        recent_months=recent_months,
        horizons=horizons,
        top_n_latest=top_n_latest,
    )


def predict_full_from_master_csv(
    master_csv_path: str | Path,
    models_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    top_n_latest: int = TOP_N_LATEST,
) -> Dict[str, Any]:
    return predict_from_master_csv(
        master_csv_path=master_csv_path,
        models_dir=models_dir,
        output_dir=output_dir,
        mode="full",
        horizons=horizons,
        top_n_latest=top_n_latest,
    )


def _normalize_symbols_input(symbols: Optional[str | Sequence[str]]) -> Optional[list[str]]:
    if symbols is None:
        return None
    if isinstance(symbols, str):
        raw_items = symbols.split(",")
    else:
        raw_items = list(symbols)

    cleaned: list[str] = []
    seen = set()
    for item in raw_items:
        if item is None:
            continue
        symbol = str(item).strip().upper()
        if symbol and symbol not in seen:
            cleaned.append(symbol)
            seen.add(symbol)
    return cleaned or None


def _add_symbol_direction_columns(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    out = df.copy()
    for horizon in horizons:
        prob_col = f"prob_up_t{horizon}"
        if prob_col in out.columns:
            prob = pd.to_numeric(out[prob_col], errors="coerce").fillna(0.5)
            out[f"direction_t{horizon}"] = np.where(prob >= 0.5, "TĂNG", "GIẢM")
    if "avg_prob_up" in out.columns:
        avg_prob = pd.to_numeric(out["avg_prob_up"], errors="coerce").fillna(0.5)
        out["direction_ensemble"] = np.where(avg_prob >= 0.5, "TĂNG", "GIẢM")
    return out

def _select_default_symbols_from_prediction_df(
    prediction_df: pd.DataFrame,
    top_n: int = 5,
) -> list[str]:
    top_n = max(int(top_n), 1)
    if prediction_df.empty:
        return []

    if "LS_GiaTriKhopLenh" in prediction_df.columns and prediction_df["LS_GiaTriKhopLenh"].notna().any():
        ranking = prediction_df.groupby("symbol")["LS_GiaTriKhopLenh"].sum().sort_values(ascending=False)
    elif "LS_KhoiLuongKhopLenh" in prediction_df.columns and prediction_df["LS_KhoiLuongKhopLenh"].notna().any():
        ranking = prediction_df.groupby("symbol")["LS_KhoiLuongKhopLenh"].sum().sort_values(ascending=False)
    else:
        ranking = prediction_df.groupby("symbol").size().sort_values(ascending=False)

    return [str(symbol).upper() for symbol in ranking.head(top_n).index.tolist()]


def _resolve_symbol_selection(
    prediction_df: pd.DataFrame,
    symbols: Optional[str | Sequence[str]] = None,
    default_top_symbols: int = DEFAULT_POPULAR_SYMBOLS,
) -> list[str]:
    available = sorted(prediction_df["symbol"].dropna().astype(str).str.upper().unique().tolist())
    available_set = set(available)
    requested = _normalize_symbols_input(symbols)

    if requested is None:
        selected = [
            symbol
            for symbol in _select_default_symbols_from_prediction_df(prediction_df, top_n=default_top_symbols)
            if symbol in available_set
        ]
    else:
        missing = [symbol for symbol in requested if symbol not in available_set]
        if missing:
            raise ValueError(
                "Một số mã không tồn tại trong cửa sổ dự đoán hiện tại: " + ", ".join(missing)
            )
        selected = requested

    if not selected:
        raise ValueError("Không xác định được mã cổ phiếu cần dự đoán.")
    return selected

def _build_symbol_direction_table(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    base_cols = [
        "Ngay", "symbol", "group", "icb_name2", "com_type_code", "LS_GiaDieuChinh",
        "avg_prob_up", "direction_ensemble",
    ]
    horizon_cols = []
    for horizon in horizons:
        for col in (f"direction_t{horizon}", f"prob_up_t{horizon}"):
            if col in df.columns:
                horizon_cols.append(col)
    use_cols = [col for col in base_cols + horizon_cols if col in df.columns]
    if not use_cols:
        return pd.DataFrame()
    return df[use_cols].copy()


def build_symbol_api_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    latest_df = result["latest_selected_symbol_table"].copy()
    if "Ngay" in latest_df.columns:
        latest_df["Ngay"] = latest_df["Ngay"].dt.strftime("%Y-%m-%d")
    return {
        "window_start": result["summary"]["window_start"],
        "window_end": result["summary"]["window_end"],
        "latest_date": result["summary"]["latest_date"],
        "selected_symbols": result["selected_symbols"],
        "predictions": latest_df.replace({np.nan: None}).to_dict(orient="records"),
    }


def _prediction_date_for_filename(result: Dict[str, Any]) -> str:
    latest_date = result.get("summary", {}).get("latest_date", "")
    if latest_date:
        try:
            return pd.to_datetime(latest_date).strftime("%d-%m-%Y")
        except Exception:
            pass
    return pd.Timestamp.today().strftime("%d-%m-%Y")


def _safe_symbol_filename(symbol: str) -> str:
    cleaned = "".join(ch for ch in str(symbol).upper() if ch.isalnum() or ch in ("_", "-"))
    return cleaned or "UNKNOWN"


def export_symbol_prediction_outputs(result: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("*.csv", "*.json", "*.png"):
        for old_file in output_dir.glob(pattern):
            old_file.unlink(missing_ok=True)

    export_date = _prediction_date_for_filename(result)
    exported_files: Dict[str, str] = {}
    selected_symbols = result.get("selected_symbols", []) or []

    for symbol in selected_symbols:
        fig = plot_single_symbol_direction(result, symbol)
        filename = f"{_safe_symbol_filename(symbol)}_Prediction_{export_date}.png"
        file_path = output_dir / filename
        fig.savefig(file_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        exported_files[str(symbol).upper()] = str(file_path)

    horizon_chart_path = output_dir / f"All_Horizons_Prediction_{export_date}.png"
    horizon_fig = plot_latest_symbol_signal(result)
    horizon_fig.savefig(horizon_chart_path, dpi=160, bbox_inches="tight")
    plt.close(horizon_fig)
    exported_files["ALL_HORIZONS"] = str(horizon_chart_path)

    return exported_files

def plot_single_symbol_direction(result: Dict[str, Any], symbol: str) -> plt.Figure:
    configure_plotting()
    selected_df = result["selected_prediction_df"].copy()
    horizons = result.get("horizons", DEFAULT_HORIZONS)
    symbol = str(symbol).upper()
    symbol_df = selected_df[selected_df["symbol"].astype(str).str.upper() == symbol].copy()
    symbol_df = symbol_df.sort_values("Ngay")
    if symbol_df.empty:
        raise ValueError(f"Kh?ng c? d? li?u cho m? {symbol} ?? v? bi?u ??.")

    fig, ax = plt.subplots(figsize=(16, 6.5))
    line_colors = {
        "T+3": "#1565C0",
        "T+7": "#2E7D32",
        "T+15": "#EF6C00",
        "T+30": "#6A1B9A",
        "Ensemble": "#263238",
    }

    ax.axhspan(0.5, 1.0, color="#E8F5E9", alpha=0.55)
    ax.axhspan(0.0, 0.5, color="#FFEBEE", alpha=0.55)
    ax.axhline(0.5, color="#455A64", linestyle="--", linewidth=1.1)

    plotted_labels = []
    for horizon in horizons:
        prob_col = f"prob_up_t{horizon}"
        if prob_col not in symbol_df.columns:
            continue
        series = pd.to_numeric(symbol_df[prob_col], errors="coerce").fillna(0.5)
        label = f"T+{horizon}"
        ax.plot(
            symbol_df["Ngay"],
            series,
            linewidth=1.9,
            alpha=0.95,
            color=line_colors.get(label, None),
            label=label,
        )
        plotted_labels.append(label)

    if "avg_prob_up" in symbol_df.columns:
        ensemble = pd.to_numeric(symbol_df["avg_prob_up"], errors="coerce").fillna(0.5)
        ax.plot(
            symbol_df["Ngay"],
            ensemble,
            linewidth=2.8,
            alpha=0.95,
            color=line_colors["Ensemble"],
            label="Ensemble",
        )
        latest_row = symbol_df.iloc[-1]
        title_tail = f"Ensemble m?i nh?t: {latest_row['direction_ensemble']} | avg_prob_up={latest_row['avg_prob_up']:.3f}"
    else:
        latest_row = symbol_df.iloc[-1]
        title_tail = "Kh?ng c? ensemble"

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("X?c su?t t?ng")
    ax.set_xlabel("Ng?y")
    ax.set_title(f"{symbol} | {title_tail}", loc="left", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    if plotted_labels or "avg_prob_up" in symbol_df.columns:
        ax.legend(ncol=min(5, len(plotted_labels) + 1), loc="upper left", frameon=True)
    plt.tight_layout()
    return fig

def plot_latest_symbol_signal(result: Dict[str, Any]) -> plt.Figure:
    configure_plotting()
    latest_df = result["latest_selected_symbol_table"].copy()
    if latest_df.empty:
        raise ValueError("Không có dữ liệu dự đoán mới nhất để vẽ biểu đồ.")

    horizons = [h for h in result.get("horizons", DEFAULT_HORIZONS) if f"prob_up_t{h}" in latest_df.columns]
    if not horizons:
        raise ValueError("Không tìm thấy cột xác suất theo horizon để vẽ biểu đồ.")

    ncols = 2
    nrows = int(np.ceil(len(horizons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(7.5, 4.8 * nrows)))
    axes = np.atleast_1d(axes).ravel()

    latest_date = result["summary"].get("latest_date", "")
    fig.suptitle(f"4 horizon chart mới nhất - {latest_date}", fontsize=16, fontweight="bold")

    for ax, horizon in zip(axes, horizons):
        prob_col = f"prob_up_t{horizon}"
        plot_df = latest_df[["symbol", prob_col]].copy()
        plot_df[prob_col] = pd.to_numeric(plot_df[prob_col], errors="coerce").fillna(0.5)
        plot_df = plot_df.sort_values(prob_col, ascending=True).reset_index(drop=True)
        colors = np.where(plot_df[prob_col] >= 0.5, "#2E7D32", "#C62828")

        bars = ax.barh(plot_df["symbol"], plot_df[prob_col], color=colors, alpha=0.9)
        ax.axvline(0.5, color="#455A64", linestyle="--", linewidth=1.1)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"Horizon T+{horizon}", loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("Xác suất tăng")
        ax.set_ylabel("Mã cổ phiếu")

        for bar, (_, row) in zip(bars, plot_df.iterrows()):
            x = min(row[prob_col] + 0.02, 0.98)
            label = "TĂNG" if row[prob_col] >= 0.5 else "GIẢM"
            ax.text(x, bar.get_y() + bar.get_height() / 2, f"{label} | {row[prob_col]:.2f}", va="center", ha="left", fontsize=9)

    for ax in axes[len(horizons):]:
        ax.set_axis_off()

    plt.tight_layout()
    return fig

def plot_symbol_direction_dashboard(result: Dict[str, Any]) -> plt.Figure:
    configure_plotting()
    selected_df = result["selected_prediction_df"].copy()
    if selected_df.empty:
        raise ValueError("Không có dữ liệu cho các mã được chọn để vẽ biểu đồ.")

    horizons = result.get("horizons", DEFAULT_HORIZONS)
    selected_symbols = result.get("selected_symbols", []) or sorted(selected_df["symbol"].unique().tolist())
    fig, axes = plt.subplots(len(selected_symbols), 1, figsize=(18, max(4.5, 4.4 * len(selected_symbols))), sharex=True)
    if len(selected_symbols) == 1:
        axes = [axes]

    line_colors = {
        "T+3": "#1565C0",
        "T+7": "#2E7D32",
        "T+15": "#EF6C00",
        "T+30": "#6A1B9A",
        "Ensemble": "#263238",
    }
    fig.suptitle("Dự đoán theo tất cả model T+3 / T+7 / T+15 / T+30", fontsize=16, fontweight="bold")

    for ax, symbol in zip(axes, selected_symbols):
        symbol_df = selected_df[selected_df["symbol"].astype(str).str.upper() == symbol].copy()
        symbol_df = symbol_df.sort_values("Ngay")
        if symbol_df.empty:
            ax.text(0.5, 0.5, f"Không có dữ liệu cho {symbol}", ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.axhspan(0.5, 1.0, color="#E8F5E9", alpha=0.55)
        ax.axhspan(0.0, 0.5, color="#FFEBEE", alpha=0.55)
        ax.axhline(0.5, color="#455A64", linestyle="--", linewidth=1.1)

        plotted_labels = []
        for horizon in horizons:
            prob_col = f"prob_up_t{horizon}"
            if prob_col not in symbol_df.columns:
                continue
            series = pd.to_numeric(symbol_df[prob_col], errors="coerce").fillna(0.5)
            label = f"T+{horizon}"
            ax.plot(
                symbol_df["Ngay"],
                series,
                linewidth=1.8,
                alpha=0.95,
                color=line_colors.get(label, None),
                label=label,
            )
            plotted_labels.append(label)

        if "avg_prob_up" in symbol_df.columns:
            ensemble = pd.to_numeric(symbol_df["avg_prob_up"], errors="coerce").fillna(0.5)
            ax.plot(
                symbol_df["Ngay"],
                ensemble,
                linewidth=2.7,
                alpha=0.95,
                color=line_colors["Ensemble"],
                label="Ensemble",
            )
            latest_row = symbol_df.iloc[-1]
            title_tail = f"Ensemble: {latest_row['direction_ensemble']} | avg_prob_up={latest_row['avg_prob_up']:.3f}"
        else:
            latest_row = symbol_df.iloc[-1]
            title_tail = "Không có ensemble"

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("XS tăng")
        ax.set_title(f"{symbol} | {title_tail}", loc="left", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        if plotted_labels or "avg_prob_up" in symbol_df.columns:
            ax.legend(ncol=min(5, len(plotted_labels) + 1), loc="upper left", frameon=True)

    axes[-1].set_xlabel("Ngày")
    plt.tight_layout()
    return fig

def predict_symbols_from_master_csv(
    master_csv_path: str | Path,
    models_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    mode: str = "recent_months",
    recent_months: int = DEFAULT_SYMBOL_LOOKBACK_MONTHS,
    recent_days: int = 22,
    score_start: Optional[str | pd.Timestamp] = None,
    score_end: Optional[str | pd.Timestamp] = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    symbols: Optional[str | Sequence[str]] = None,
    default_top_symbols: int = DEFAULT_POPULAR_SYMBOLS,
) -> Dict[str, Any]:
    base_result = predict_from_master_csv(
        master_csv_path=master_csv_path,
        models_dir=models_dir,
        output_dir=None,
        mode=mode,
        recent_months=recent_months,
        recent_days=recent_days,
        score_start=score_start,
        score_end=score_end,
        horizons=horizons,
    )

    prediction_df = _add_symbol_direction_columns(base_result["prediction_df"], horizons=horizons)
    latest_date = prediction_df["Ngay"].max()
    selected_symbols = _resolve_symbol_selection(
        prediction_df=prediction_df,
        symbols=symbols,
        default_top_symbols=default_top_symbols,
    )

    selected_prediction_df = prediction_df[
        prediction_df["symbol"].astype(str).str.upper().isin(selected_symbols)
    ].copy()
    latest_selected_df = selected_prediction_df[selected_prediction_df["Ngay"] == latest_date].copy()

    selected_history_table = _build_symbol_direction_table(selected_prediction_df, horizons)
    latest_selected_symbol_table = (
        _build_symbol_direction_table(latest_selected_df, horizons)
        .sort_values(["avg_prob_up", "symbol"], ascending=[False, True])
        .reset_index(drop=True)
    )

    result = dict(base_result)
    result["prediction_df"] = prediction_df
    result["selected_symbols"] = selected_symbols
    result["selected_prediction_df"] = selected_prediction_df.sort_values(["symbol", "Ngay"]).reset_index(drop=True)
    result["latest_selected_df"] = latest_selected_df.sort_values(["avg_prob_up", "symbol"], ascending=[False, True]).reset_index(drop=True)
    result["selected_history_table"] = selected_history_table.sort_values(["symbol", "Ngay"]).reset_index(drop=True)
    result["latest_selected_symbol_table"] = latest_selected_symbol_table
    result["summary"] = dict(result["summary"])
    result["summary"]["selected_symbols"] = selected_symbols
    result["summary"]["selected_symbols_count"] = len(selected_symbols)
    result["horizons"] = tuple(horizons)
    result["symbol_api_payload"] = build_symbol_api_payload(result)

    if output_dir is not None:
        result["symbol_exported_files"] = export_symbol_prediction_outputs(result, output_dir=output_dir)
    else:
        result["symbol_exported_files"] = {}

    return result

def predict_default_symbols(
    master_csv_path: str | Path,
    models_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    recent_months: int = DEFAULT_SYMBOL_LOOKBACK_MONTHS,
    default_top_symbols: int = DEFAULT_POPULAR_SYMBOLS,
    mode: str = "recent_months",
) -> Dict[str, Any]:
    """API đơn giản nhất: tự chọn các mã phổ biến nhất và dự đoán tăng/giảm."""
    return predict_symbols_from_master_csv(
        master_csv_path=master_csv_path,
        models_dir=models_dir,
        output_dir=output_dir,
        mode=mode,
        recent_months=recent_months,
        symbols=None,
        default_top_symbols=default_top_symbols,
    )

def predict_selected_symbols(
    master_csv_path: str | Path,
    models_dir: str | Path,
    symbols: str | Sequence[str],
    output_dir: Optional[str | Path] = None,
    recent_months: int = DEFAULT_SYMBOL_LOOKBACK_MONTHS,
    mode: str = "recent_months",
) -> Dict[str, Any]:
    """API gọn cho trường hợp đã biết rõ một hoặc nhiều mã cần dự đoán."""
    return predict_symbols_from_master_csv(
        master_csv_path=master_csv_path,
        models_dir=models_dir,
        output_dir=output_dir,
        mode=mode,
        recent_months=recent_months,
        symbols=symbols,
    )

def predict_one_symbol(
    master_csv_path: str | Path,
    models_dir: str | Path,
    symbol: str,
    output_dir: Optional[str | Path] = None,
    recent_months: int = DEFAULT_SYMBOL_LOOKBACK_MONTHS,
    mode: str = "recent_months",
) -> Dict[str, Any]:
    """API gọn nhất cho một mã cổ phiếu duy nhất."""
    return predict_selected_symbols(
        master_csv_path=master_csv_path,
        models_dir=models_dir,
        symbols=[symbol],
        output_dir=output_dir,
        recent_months=recent_months,
        mode=mode,
    )

def integration_examples(root_dir: str | Path = "..") -> str:
    root_dir = str(root_dir)
    return f"""Ví dụ gọi hàm từ project khác
--------------------------------
1. Cách khuyến nghị nhất cho user cuối: dự đoán theo từng mã cổ phiếu

from test_model.model_apply_api import predict_symbols_from_master_csv

result = predict_symbols_from_master_csv(
    master_csv_path=r"{root_dir}/dataset.csv",
    models_dir=r"{root_dir}/models",
    output_dir=r"{root_dir}/test_model/output",
    mode="recent_months",
    recent_months={DEFAULT_SYMBOL_LOOKBACK_MONTHS},
    symbols=None,
    default_top_symbols={DEFAULT_POPULAR_SYMBOLS},
)

latest_symbol_table = result["latest_selected_symbol_table"]
symbol_payload = result["symbol_api_payload"]

2. Nếu muốn chỉ định rõ một hoặc nhiều mã:

from test_model.model_apply_api import predict_symbols_from_master_csv

result = predict_symbols_from_master_csv(
    master_csv_path=r"{root_dir}/dataset.csv",
    models_dir=r"{root_dir}/models",
    output_dir=r"{root_dir}/test_model/output",
    mode="recent_months",
    recent_months={DEFAULT_SYMBOL_LOOKBACK_MONTHS},
    symbols=["VCB", "BID", "CTG"],
)

3. Nếu muốn dùng toàn bộ dataset:

from test_model.model_apply_api import predict_symbols_from_master_csv

result = predict_symbols_from_master_csv(
    master_csv_path=r"{root_dir}/dataset.csv",
    models_dir=r"{root_dir}/models",
    output_dir=r"{root_dir}/test_model/output",
    mode="full",
    symbols=["VCB"],
)

4. Nếu vẫn cần luồng output đầy đủ cho tất cả mã:

from test_model.model_apply_api import predict_recent_from_master_csv

result = predict_recent_from_master_csv(
    master_csv_path=r"{root_dir}/dataset.csv",
    models_dir=r"{root_dir}/models",
    output_dir=r"{root_dir}/test_model/output",
    recent_months={DEFAULT_SYMBOL_LOOKBACK_MONTHS},
)

latest_table = result["latest_tables"]["ensemble"]
api_payload = result["api_payload"]

5. Nếu muốn chạy như một lệnh dòng lệnh:

python model_apply_api.py ^
  --input ../dataset.csv ^
  --models ../models ^
  --output ./output ^
  --mode recent_months ^
  --recent-months {DEFAULT_SYMBOL_LOOKBACK_MONTHS} ^
  --symbols VCB,BID,CTG

6. Dữ liệu mới chỉ có 1 tháng gần nhất vẫn cần lịch sử dài hơn để tính feature:
- MA50 cần khoảng 50 phiên.
- Cycle feature dài nhất cần khoảng 10 x 20 = 200 phiên.
- Vì vậy nên giữ ít nhất {MIN_LOOKBACK_TRADING_DAYS} phiên lịch sử trước cửa sổ cần dự đoán.
"""

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Áp dụng model VN100 đã train cho một file CSV master. "
            "Người dùng chỉ cần đưa vào 1 file dữ liệu thô có lịch sử từ quá khứ đến hiện tại."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn tới file CSV master đầu vào, ví dụ ../dataset.csv",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Đường dẫn tới thư mục chứa model, ví dụ ../models",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Thư mục output để lưu file kết quả. Nếu bỏ trống thì không export file.",
    )
    parser.add_argument(
        "--mode",
        default="recent_months",
        choices=["latest_day", "recent_months", "recent_days", "date_range", "full"],
        help="Chế độ chọn khoảng thời gian cần chấm điểm.",
    )
    parser.add_argument(
        "--recent-months",
        type=int,
        default=DEFAULT_SYMBOL_LOOKBACK_MONTHS,
        help=f"Số tháng gần nhất khi dùng mode recent_months. Mặc định là {DEFAULT_SYMBOL_LOOKBACK_MONTHS} tháng.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=22,
        help="Số phiên gần nhất khi dùng mode recent_days.",
    )
    parser.add_argument(
        "--score-start",
        default=None,
        help="Ngày bắt đầu khi dùng mode date_range, ví dụ 2026-03-01",
    )
    parser.add_argument(
        "--score-end",
        default=None,
        help="Ngày kết thúc khi dùng mode date_range, ví dụ 2026-03-29",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Danh s?ch m? c? phi?u, ng?n c?ch b?i d?u ph?y. V? d?: VCB,BID,CTG. N?u b? tr?ng s? t? ch?n 10 m? thanh kho?n cao nh?t.",
    )
    parser.add_argument(
        "--default-top-symbols",
        type=int,
        default=DEFAULT_POPULAR_SYMBOLS,
        help=f"Số mã mặc định cần tự chọn khi không truyền --symbols. Mặc định là {DEFAULT_POPULAR_SYMBOLS}.",
    )
    return parser

def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    result = predict_symbols_from_master_csv(
        master_csv_path=args.input,
        models_dir=args.models,
        output_dir=args.output,
        mode=args.mode,
        recent_months=args.recent_months,
        recent_days=args.recent_days,
        score_start=args.score_start,
        score_end=args.score_end,
        symbols=args.symbols,
        default_top_symbols=args.default_top_symbols,
    )

    print("\nTÓM TẮT CHẠY MODEL THEO MÃ")
    print("- Cửa sổ dự đoán:", result["summary"]["window_start"], "->", result["summary"]["window_end"])
    print("- Ngày mới nhất  :", result["summary"]["latest_date"])
    print("- Mã được chọn   :", ", ".join(result["selected_symbols"]))
    print("- Số phiên       :", result["summary"]["trading_days_scored"])
    print("- Lookback warn  :", result["summary"]["lookback_warning"])

    latest_table = result["latest_selected_symbol_table"].copy()
    print("\nDỰ ĐOÁN TĂNG / GIẢM MỚI NHẤT")
    print(latest_table.to_string(index=False))

    if result.get("symbol_exported_files"):
        print("\nFILE ĐÃ XUẤT")
        for name, path in result["symbol_exported_files"].items():
            print(f"- {name}: {path}")

    return 0

__all__ = [
    "BUY_THRESHOLD",
    "AVOID_THRESHOLD",
    "DEFAULT_HORIZONS",
    "DEFAULT_POPULAR_SYMBOLS",
    "DEFAULT_SYMBOL_LOOKBACK_MONTHS",
    "MIN_LOOKBACK_TRADING_DAYS",
    "add_realized_outcomes",
    "build_api_payload",
    "build_feature_table",
    "build_symbol_api_payload",
    "cli_main",
    "clean_raw_dataframe",
    "combine_history_and_recent",
    "configure_plotting",
    "export_prediction_outputs",
    "export_symbol_prediction_outputs",
    "integration_examples",
    "load_and_clean_csv",
    "load_model_bundle",
    "make_summary_table",
    "plot_latest_day_dashboard",
    "plot_latest_symbol_signal",
    "plot_prediction_overview",
    "plot_realized_quality",
    "plot_single_symbol_direction",
    "plot_symbol_direction_dashboard",
    "predict_default_symbols",
    "predict_from_clean_data",
    "predict_from_csv",
    "predict_full_from_master_csv",
    "predict_from_master_csv",
    "predict_latest_from_master_csv",
    "predict_one_symbol",
    "predict_recent_from_master_csv",
    "predict_selected_symbols",
    "predict_symbols_from_master_csv",
]

if __name__ == "__main__":
    raise SystemExit(cli_main())
