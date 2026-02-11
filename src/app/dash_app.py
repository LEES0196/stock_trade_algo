"""Dash (Plotly) user interface for the auto portfolio strategy.

This module defines a Dash application allowing users to configure and run
the momentum‐based portfolio strategy defined in ``src/strategy``. Users can
modify the YAML configuration in a textarea, click a button to execute the
strategy, and view the resulting allocations in a table. Each run is appended
to the CSV file specified in the configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Any

import yaml
import pandas as pd
import numpy as np
try:
    import plotly.graph_objs as go  # type: ignore
except Exception:  # fallback if plotly missing
    go = None  # type: ignore

# Attempt to import Dash; if it fails, write to logs/error.log and re-raise
try:
    from dash import (
        Dash,
        html,
        dcc,
        dash_table,
        Input,
        Output,
        State,
        no_update,
        callback_context,
    )
except Exception as _dash_import_exc:
    import traceback, datetime

    try:
        _logs_dir = Path(__file__).resolve().parents[2] / "logs"
        _logs_dir.mkdir(parents=True, exist_ok=True)
        _log_path = _logs_dir / "error.log"
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        with _log_path.open("a", encoding="utf-8") as _f:
            _f.write(f"[{ts}] ImportError in dash_app: {_dash_import_exc!r}\n")
            _f.write(traceback.format_exc() + "\n")
    except Exception:
        # Ignore logging failures in read-only environments
        pass
    raise

from ..strategy.allocate import run_all
from ..strategy.io import append_decision
from ..strategy.data import download_prices
from ..strategy.nn_predictor import (
    train_or_load_model,
    predict_probabilities,
    _parse_time_interval,
)
 

# Friendly display names for tickers
_FRIENDLY_TICKERS = {
    "294400.ks": "KOSEF 200",
    "148070.ks": "KOSEF 10yr KTB",
}

def _friendly_ticker_name(t: str) -> str:
    try:
        return _FRIENDLY_TICKERS.get(t.lower(), t)
    except Exception:
        return t

# Default path to the sample configuration
CFG_PATH = Path(__file__).resolve().parents[2] / "config" / "base.yaml"


def run_strategy(config_yaml: str) -> pd.DataFrame:
    """Parse YAML, run the strategy, append decisions and return results.

    Returns a DataFrame suitable for display with tickers in index.
    """
    cfg = yaml.safe_load(config_yaml)
    df = run_all(cfg)
    # Append decision to CSV if specified in config
    io_cfg = cfg.get("io", {})
    csv_path = io_cfg.get("decisions_csv")
    if csv_path:
        try:
            append_decision(df, csv_path)
        except Exception as e:
            # Log non-fatal persistence errors
            import traceback, datetime

            logs_dir = Path(__file__).resolve().parents[2] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "error.log"
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{ts}] Error appending decision: {e!r}\n")
                f.write(traceback.format_exc() + "\n")
    return df


def _coerce_tickers(s: str) -> List[str]:
    return [t.strip() for t in (s or "").split(",") if t.strip()]


def build_cfg_from_controls(values: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a configuration dict (compatible with YAML schema) from UI values."""
    # Base defaults from file to keep unspecified keys stable
    try:
        base = yaml.safe_load(CFG_PATH.read_text()) or {}
    except Exception:
        base = {}
    # Overwrite with form values
    base.setdefault("data", {})
    base["data"].update({
        "period": values.get("data_period", "5y"),
        "interval": values.get("data_interval", "1d"),
    })

    base.setdefault("universe", {})
    base["universe"].update({
        "aggressive": _coerce_tickers(values.get("uni_agg", "")),
        "passive": _coerce_tickers(values.get("uni_pas", "")),
        "score_set": _coerce_tickers(values.get("uni_score", "")),
        "cash_ticker": values.get("uni_cash", "CASH"),
    })

    base.setdefault("features", {})
    base["features"].update({
        "months": int(values.get("feat_months", 12)),
        "rebalance_day": values.get("feat_rebalance", "EOM"),
    })

    base.setdefault("allocation", {})
    a1 = base["allocation"].setdefault("algo1", {})
    a1.update({
        "passive_top_n": int(values.get("a1_topn", 3)),
        "passive_rank_col": values.get("a1_rank", "6M"),
        "aggressive_pos_col": values.get("a1_aggr", "12M"),
    })
    a2 = base["allocation"].setdefault("algo2", {})
    a2.update({
        "cash_score": float(values.get("a2_cash", 0.15)),
    })
    blend = base["allocation"].setdefault("blend", {})
    w1 = float(values.get("blend_w1", 0.6))
    blend.update({
        "w1": w1,
        "w2": max(0.0, min(1.0, 1.0 - w1)),
    })
    caps = base["allocation"].setdefault("caps", {})
    caps.update({
        "max_weight": float(values.get("caps_max", 0.8)),
        "min_weight": float(values.get("caps_min", 0.0)),
        "cash_cap": float(values.get("caps_cash", 0.7)),
    })

    base.setdefault("io", {})
    if values.get("io_csv"):
        base["io"]["decisions_csv"] = values.get("io_csv")

    # Model section (optional)
    mtype = values.get("model_type", "").strip().lower()
    if mtype:
        base.setdefault("model", {})
        base["model"].update({
            "type": mtype,
            "lookback_window": int(values.get("model_lookback", 60)),
            "time_interval": values.get("model_time", "5d"),
            "confidence_interval": float(values.get("model_conf", 0.7)),
            "train": bool(values.get("model_train", False)),
        })
        if values.get("model_path"):
            base["model"]["model_path"] = values.get("model_path")

    return base


def default_config() -> str:
    """Load the default YAML configuration from the repository."""
    try:
        return CFG_PATH.read_text()
    except Exception:
        return ""


def _df_to_table_payload(df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    """Convert a DataFrame to Dash DataTable data and columns payload."""
    if df is None or df.empty:
        return [], []
    # Apply friendly names to the index for display if index holds tickers
    try:
        df_disp = df.copy()
        df_disp.index = [
            _friendly_ticker_name(ix) if isinstance(ix, str) else ix for ix in df_disp.index
        ]
    except Exception:
        df_disp = df
    df_out = df_disp.reset_index()
    columns = [{"name": c, "id": c} for c in df_out.columns]
    data = df_out.to_dict("records")
    return data, columns


def create_app() -> Dash:
    """Construct and return a Dash app."""
    assets_path = str((Path(__file__).resolve().parents[2] / "assets").resolve())
    app = Dash(__name__, title="Momentum Blend Strategy", assets_folder=assets_path)

    # Load defaults to seed controls
    try:
        _defaults = yaml.safe_load(default_config()) or {}
    except Exception:
        _defaults = {}
    _d = _defaults
    _data = _d.get("data", {})
    _uni = _d.get("universe", {})
    _feat = _d.get("features", {})
    _alloc = _d.get("allocation", {})
    _a1 = _alloc.get("algo1", {})
    _a2 = _alloc.get("algo2", {})
    _blend = _alloc.get("blend", {})
    _caps = _alloc.get("caps", {})
    _io = _d.get("io", {})
    _model = _d.get("model", {})

    app.layout = html.Div(
        className="app-container",
        children=[
            html.Div(
                className="header",
                children=[
                    html.Div([
                        html.H1("Momentum Strategy"),
                        html.P("Aggressive/Passive blend with optional NN predictions"),
                    ], className="header-text"),
                ],
                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
            ),
            dcc.Tabs(id="tabs", value="tab-strategy", className="dash-tabs", children=[
                dcc.Tab(label="Strategy", value="tab-strategy", children=[
                    html.Div(
                        className="grid-2",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.H3("Form Controls", className="section-title"),
                                    html.Div([
                                        html.H4("Data"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Period"),
                                                dcc.Dropdown(
                                                    id="f-data-period",
                                                    options=[{"label": p, "value": p} for p in ["1y", "3y", "5y", "10y", "max"]],
                                                    value=_data.get("period", "5y"),
                                                    clearable=False,
                                                    style={"width": "100%"},
                                                ),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Interval"),
                                                dcc.Dropdown(
                                                    id="f-data-interval",
                                                    options=[{"label": i, "value": i} for i in ["1d", "1wk", "1mo"]],
                                                    value=_data.get("interval", "1d"),
                                                    clearable=False,
                                                    style={"width": "100%"},
                                                ),
                                            ], className="form-control"),
                                        ], className="form-grid form-2"),

                                        html.H4("Universe"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Aggressive (comma-separated)"),
                                                dcc.Input(id="f-uni-agg", type="text", value=", ".join(_uni.get("aggressive", [])), className="full-width"),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Passive (comma-separated)"),
                                                dcc.Input(id="f-uni-pas", type="text", value=", ".join(_uni.get("passive", [])), className="full-width"),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Score set (comma-separated)"),
                                                dcc.Input(id="f-uni-score", type="text", value=", ".join(_uni.get("score_set", [])), className="full-width"),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Cash ticker"),
                                                dcc.Input(id="f-uni-cash", type="text", value=_uni.get("cash_ticker", "CASH"), style={"width": "200px"}),
                                            ], className="form-control"),
                                        ]),

                                        html.H4("Features"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Months"),
                                                dcc.Input(id="f-feat-months", type="number", value=int(_feat.get("months", 12)), min=1, step=1, style={"width": "100%"}),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Rebalance"),
                                                dcc.Dropdown(
                                                    id="f-feat-rebalance",
                                                    options=[{"label": v, "value": v} for v in ["EOM", "BOM"]],
                                                    value=_feat.get("rebalance_day", "EOM"),
                                                    clearable=False,
                                                    style={"width": "100%"},
                                                ),
                                            ], className="form-control"),
                                        ], className="form-grid form-2"),

                                        html.H4("Allocation — Algo 1"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Passive top N"),
                                                dcc.Input(id="f-a1-topn", type="number", value=int(_a1.get("passive_top_n", 3)), min=1, step=1, style={"width": "100%"}),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Rank column"),
                                                dcc.Dropdown(
                                                    id="f-a1-rank",
                                                    options=[{"label": c, "value": c} for c in ["3M", "6M", "12M"]],
                                                    value=_a1.get("passive_rank_col", "6M"),
                                                    clearable=False,
                                                    style={"width": "100%"},
                                                ),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Aggressive positive col"),
                                                dcc.Dropdown(
                                                    id="f-a1-aggr",
                                                    options=[{"label": c, "value": c} for c in ["6M", "12M"]],
                                                    value=_a1.get("aggressive_pos_col", "12M"),
                                                    clearable=False,
                                                    style={"width": "100%"},
                                                ),
                                            ], className="form-control"),
                                        ], className="form-grid form-3"),

                                        html.H4("Allocation — Algo 2"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Cash score"),
                                                dcc.Input(id="f-a2-cash", type="number", value=float(_a2.get("cash_score", 0.15)), min=0.0, max=1.0, step=0.01, style={"width": "100%"}),
                                            ], className="form-control"),
                                        ]),

                                        html.H4("Blend & Caps"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Blend w1 (algo1)"),
                                                dcc.Slider(id="f-blend-w1", min=0.0, max=1.0, step=0.05, value=float(_blend.get("w1", 0.6)), tooltip={"always_visible": False}, className="dash-slider"),
                                                html.Div(f"w2 (algo2) = {1.0 - float(_blend.get('w1', 0.6)):.2f}", id="f-blend-w2-label", className="muted"),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Max weight"),
                                                dcc.Input(id="f-caps-max", type="number", value=float(_caps.get("max_weight", 0.8)), min=0.0, max=1.0, step=0.05, style={"width": "100%"}),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Min weight"),
                                                dcc.Input(id="f-caps-min", type="number", value=float(_caps.get("min_weight", 0.0)), min=0.0, max=1.0, step=0.05, style={"width": "100%"}),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Cash cap"),
                                                dcc.Input(id="f-caps-cash", type="number", value=float(_caps.get("cash_cap", 0.7)), min=0.0, max=1.0, step=0.05, style={"width": "100%"}),
                                            ], className="form-control"),
                                        ], className="form-grid form-3"),

                                        html.H4("Model (optional)"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Type"),
                                                dcc.Dropdown(
                                                    id="f-model-type",
                                                    options=[
                                                        {"label": "(disabled)", "value": ""},
                                                        {"label": "Neural (NN)", "value": "nn"},
                                                    ],
                                                    value=str(_model.get("type", "")),
                                                    clearable=False,
                                                    style={"width": "100%"},
                                                ),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Lookback"),
                                                dcc.Input(id="f-model-lookback", type="number", value=int(_model.get("lookback_window", 60)), min=10, step=1, style={"width": "100%"}),
                                            ], className="form-control"),
                                            html.Div([
                                                html.Label("Time interval"),
                                                dcc.Input(id="f-model-time", type="text", value=_model.get("time_interval", "5d"), style={"width": "100%"}),
                                            ], className="form-control"),
                                        ], className="form-grid form-3"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Confidence"),
                                                dcc.Slider(
                                                    id="f-model-conf",
                                                    min=0.0,
                                                    max=1.0,
                                                    step=0.05,
                                                    value=float(_model.get("confidence_interval", 0.7)),
                                                    tooltip={"always_visible": False},
                                                    className="dash-slider",
                                                ),
                                            ], className="form-control"),
                                        ]),
                                        html.Div([
                                            html.Div([
                                                html.Label("Train"),
                                                dcc.Checklist(id="f-model-train", options=[{"label": "", "value": "train"}], value=["train"] if _model.get("train", False) else [] , inline=True,
                                                               style={"display": "inline-block", "verticalAlign": "middle"}),
                                            ], className="form-control"),
                                        ]),
                                        html.Div([
                                            html.Div([
                                                html.Label("Model path"),
                                                dcc.Input(id="f-model-path", type="text", value=_model.get("model_path", "data/models/nn_model.pkl"), className="full-width"),
                                            ], className="form-control"),
                                        ]),

                                        html.H4("Output"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Decisions CSV"),
                                                dcc.Input(id="f-io-csv", type="text", value=_io.get("decisions_csv", "data/outputs/decisions.csv"), className="full-width"),
                                            ], className="form-control"),
                                        ]),
                                        html.Button("Run (Form)", id="run-form-btn", n_clicks=0, className="btn btn-primary", style={"marginTop": "8px"}),
                                    ])
                                ],
                            ),
                            html.Details([
                                html.Summary("YAML Configuration", style={"cursor": "pointer", "fontWeight": 600}),
                                html.Div(
                                    className="card",
                                    children=[
                                        dcc.Textarea(
                                            id="config-box",
                                            value=default_config(),
                                            className="code-box",
                                            style={"height": "420px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Button("Run from YAML", id="run-btn", n_clicks=0, className="btn btn-outline"),
                                                html.Div(id="message", className="muted", style={"marginTop": "8px"}),
                                            ]
                                        ),
                                    ],
                                ),
                            ], open=False),
                        ],
                    ),
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Results", className="section-title"),
                            dash_table.DataTable(
                                id="out-table",
                                data=[],
                                columns=[],
                                sort_action="native",
                                page_size=20,
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                style_header={"backgroundColor": "var(--table-header)", "color": "var(--text)", "fontWeight": "600", "border": "1px solid var(--border)"},
                                style_cell={"color": "var(--text)", "backgroundColor": "var(--card-bg)", "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace", "padding": "10px", "border": "1px solid var(--border)"},
                                style_data_conditional=[
                                    {"if": {"row_index": "odd"}, "backgroundColor": "var(--table-stripe)"},
                                ],
                            ),
                        ],
                    ),
                ]),
                dcc.Tab(label="Prediction", value="tab-predict", children=[
                    html.Div(
                        className="card",
                        children=[
                            html.H3("Single‑Ticker NN Prediction", className="section-title"),
                            html.Div(
                                className="controls-row",
                                children=[
                                    html.Div([html.Label("Ticker"), dcc.Input(id="pred-ticker", type="text", value="NVDA", debounce=True)], className="control"),
                                    html.Div([html.Label("Period"), dcc.Input(id="pred-period", type="text", value="5y")], className="control"),
                                    html.Div([html.Label("Interval"), dcc.Input(id="pred-interval", type="text", value="1d")], className="control"),
                                    html.Div([html.Label("Time horizon"), dcc.Input(id="pred-time", type="text", value="5d")], className="control"),
                                    html.Div([
                                        html.Label("History window"),
                                        dcc.Input(
                                            id="pred-history",
                                            type="text",
                                            value="60d",
                                            placeholder="e.g. 60d",
                                            style={"width": "140px"},
                                        ),
                                    ], className="control control-inline"),
                                    html.Div([html.Label("Lookback"), dcc.Input(id="pred-lookback", type="number", value=60, min=10, max=500, step=1)], className="control"),
                                    html.Div([html.Label("Confidence"), dcc.Input(id="pred-conf", type="number", value=0.7, min=0.0, max=1.0, step=0.05)], className="control"),
                                    html.Div([html.Label("Train"), dcc.Checklist(id="pred-train", options=[{"label": "", "value": "train"}], value=[], inline=True)], className="control"),
                                    html.Div([html.Button("Predict", id="predict-btn", n_clicks=0, className="btn btn-primary")], className="control"),
                                ],
                            ),
                            html.Div(id="pred-message", className="muted", style={"marginTop": "8px"}),
                            dash_table.DataTable(
                                id="pred-table",
                                data=[],
                                columns=[],
                                sort_action="native",
                                page_size=5,
                                style_as_list_view=True,
                                style_table={"marginTop": "8px", "overflowX": "auto"},
                                style_header={"backgroundColor": "var(--table-header)", "color": "var(--text)", "fontWeight": "600", "border": "1px solid var(--border)"},
                                style_cell={"color": "var(--text)", "backgroundColor": "var(--card-bg)", "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace", "padding": "10px", "border": "1px solid var(--border)"},
                                style_data_conditional=[
                                    {"if": {"row_index": "odd"}, "backgroundColor": "var(--table-stripe)"},
                                ],
                            ),
                            dcc.Graph(id="pred-forecast", figure={}, style={"marginTop": "8px"}),
                        ],
                    ),
                ]),
            ]),
        ],
    )

    @app.callback(
        Output("out-table", "data"),
        Output("out-table", "columns"),
        Output("message", "children"),
        Input("run-btn", "n_clicks"),
        Input("run-form-btn", "n_clicks"),
        State("config-box", "value"),
        # Form states
        State("f-data-period", "value"),
        State("f-data-interval", "value"),
        State("f-uni-agg", "value"),
        State("f-uni-pas", "value"),
        State("f-uni-score", "value"),
        State("f-uni-cash", "value"),
        State("f-feat-months", "value"),
        State("f-feat-rebalance", "value"),
        State("f-a1-topn", "value"),
        State("f-a1-rank", "value"),
        State("f-a1-aggr", "value"),
        State("f-a2-cash", "value"),
        State("f-blend-w1", "value"),
        State("f-caps-max", "value"),
        State("f-caps-min", "value"),
        State("f-caps-cash", "value"),
        State("f-io-csv", "value"),
        State("f-model-type", "value"),
        State("f-model-lookback", "value"),
        State("f-model-time", "value"),
        State("f-model-conf", "value"),
        State("f-model-train", "value"),
        State("f-model-path", "value"),
        prevent_initial_call=True,
    )
    def on_run(n_clicks_yaml, n_clicks_form, cfg_text,
               data_period, data_interval, uni_agg, uni_pas, uni_score, uni_cash,
               feat_months, feat_rebalance, a1_topn, a1_rank, a1_aggr,
               a2_cash, blend_w1, caps_max, caps_min, caps_cash, io_csv,
               model_type, model_lookback, model_time, model_conf, model_train, model_path):  # type: ignore[unused-argument]
        which = None
        try:
            ctx = callback_context
            if ctx and ctx.triggered:
                which = ctx.triggered[0]["prop_id"].split(".")[0]
        except Exception:
            which = None
        # YAML-driven run
        if which == "run-btn":
            if not cfg_text:
                return [], [], "Please provide a YAML configuration."
            try:
                df = run_strategy(cfg_text)
            except Exception as e:
                # Log error to file for post-mortem
                import traceback, datetime

                logs_dir = Path(__file__).resolve().parents[2] / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs_dir / "error.log"
                ts = datetime.datetime.utcnow().isoformat() + "Z"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"[{ts}] Callback error: {e!r}\n")
                    f.write(traceback.format_exc() + "\n")
                return [], [], f"Error: {e}"
            data, columns = _df_to_table_payload(df)
            msg = f"Run completed (YAML). {len(data)} rows appended/displayed."
            return data, columns, msg

        # Form-driven run
        values = {
            "data_period": data_period,
            "data_interval": data_interval,
            "uni_agg": uni_agg,
            "uni_pas": uni_pas,
            "uni_score": uni_score,
            "uni_cash": uni_cash,
            "feat_months": feat_months,
            "feat_rebalance": feat_rebalance,
            "a1_topn": a1_topn,
            "a1_rank": a1_rank,
            "a1_aggr": a1_aggr,
            "a2_cash": a2_cash,
            "blend_w1": blend_w1,
            "caps_max": caps_max,
            "caps_min": caps_min,
            "caps_cash": caps_cash,
            "io_csv": io_csv,
            "model_type": model_type,
            "model_lookback": model_lookback,
            "model_time": model_time,
            "model_conf": model_conf,
            "model_train": ("train" in (model_train or [])),
            "model_path": model_path,
        }
        try:
            cfg = build_cfg_from_controls(values)
            df = run_all(cfg)
            # Append decisions using the same logic as run_strategy
            io_cfg = cfg.get("io", {})
            csv_path = io_cfg.get("decisions_csv")
            if csv_path:
                try:
                    append_decision(df, csv_path)
                except Exception as e:
                    import traceback, datetime
                    logs_dir = Path(__file__).resolve().parents[2] / "logs"
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    log_path = logs_dir / "error.log"
                    ts = datetime.datetime.utcnow().isoformat() + "Z"
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"[{ts}] Error appending decision (form): {e!r}\n")
                        f.write(traceback.format_exc() + "\n")
        except Exception as e:
            # Log error to file for post-mortem
            import traceback, datetime

            logs_dir = Path(__file__).resolve().parents[2] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "error.log"
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{ts}] Form callback error: {e!r}\n")
                f.write(traceback.format_exc() + "\n")
            return [], [], f"Error: {e}"

        data, columns = _df_to_table_payload(df)
        msg = f"Run completed (Form). {len(data)} rows appended/displayed."
        return data, columns, msg
        try:
            df = run_strategy(cfg_text)
        except Exception as e:
            # Log error to file for post-mortem
            import traceback, datetime

            logs_dir = Path(__file__).resolve().parents[2] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "error.log"
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{ts}] Callback error: {e!r}\n")
                f.write(traceback.format_exc() + "\n")
            return no_update, no_update, f"Error: {e}"

        data, columns = _df_to_table_payload(df)
        msg = f"Run completed. {len(data)} rows appended/displayed."
        return data, columns, msg

    # Live update for blend w2 label for better UX
    @app.callback(
        Output("f-blend-w2-label", "children"),
        Input("f-blend-w1", "value"),
    )
    def _update_w2_label(w1):  # type: ignore[unused-argument]
        try:
            w1f = float(w1 or 0.0)
        except Exception:
            w1f = 0.0
        w2 = max(0.0, min(1.0, 1.0 - w1f))
        return f"w2 (algo2) = {w2:.2f}"

    # Dark theme is default; no theme/restart controls

    @app.callback(
        Output("pred-table", "data"),
        Output("pred-table", "columns"),
        Output("pred-message", "children"),
        Output("pred-forecast", "figure"),
        Input("predict-btn", "n_clicks"),
        State("pred-ticker", "value"),
        State("pred-period", "value"),
        State("pred-interval", "value"),
        State("pred-time", "value"),
        State("pred-history", "value"),
        State("pred-lookback", "value"),
        State("pred-conf", "value"),
        State("pred-train", "value"),
        prevent_initial_call=True,
    )
    def on_predict(n_clicks, ticker, period, interval, time_h, history_win, lookback, conf, train_vals):  # type: ignore[unused-argument]
        if not ticker:
            return [], [], "Please provide a ticker.", {}
        try:
            t = str(ticker).upper().strip()
            period = str(period or "5y")
            interval = str(interval or "1d")
            time_h = str(time_h or "5d")
            history_win = str(history_win or "60d")
            lookback = int(lookback or 60)
            conf = float(conf or 0.7)
            train = "train" in (train_vals or [])

            prices = download_prices([t], period=period, interval=interval)
            horizon_steps = _parse_time_interval(time_h, interval)
            model = train_or_load_model(
                prices,
                lookback_window=lookback,
                horizon_steps=horizon_steps,
                model_path=None,  # do not persist from the app by default
                train=train,
                epochs=40,
            )
            probs = predict_probabilities(model, prices, lookback_window=lookback)
            p = float(probs.get(t, 0.0))
            favorable = p >= conf
            t_disp = _friendly_ticker_name(t)
            msg = (
                f"{t_disp}: P(positive {time_h}) = {p:.4f} | threshold={conf:.2f} → "
                + ("FAVORABLE" if favorable else "UNFAVORABLE")
            )
            df = pd.DataFrame(
                {
                    "ticker": [t],
                    "period": [period],
                    "interval": [interval],
                    "time_horizon": [time_h],
                    "lookback": [lookback],
                    "prob_positive": [round(p, 6)],
                    "decision": ["FAVORABLE" if favorable else "UNFAVORABLE"],
                }
            )
            data, cols = _df_to_table_payload(df.set_index("ticker"))
            # Build estimated movement via GBM Monte Carlo using recent stats
            s = prices[t].dropna().astype(float)
            S0 = float(s.iloc[-1]) if not s.empty else 0.0
            lr = np.log(s.values[1:] / s.values[:-1]) if len(s) > 1 else np.array([0.0])
            if lr.size == 0:
                figure = {}
            else:
                mu = float(lr[-min(len(lr), max(lookback, 20)):].mean())
                sigma = float(lr[-min(len(lr), max(lookback, 20)):].std() + 1e-8)
                steps = max(1, int(horizon_steps))
                sims = 500
                Z = np.random.standard_normal(size=(sims, steps))
                # GBM log-return per step: mu - 0.5*sigma^2 + sigma*Z
                incr = mu - 0.5 * sigma * sigma + sigma * Z
                log_paths = np.cumsum(incr, axis=1)
                paths = S0 * np.exp(log_paths)
                mean_path = paths.mean(axis=0)
                alpha = (1.0 - max(0.01, min(conf, 0.99))) / 2.0
                lower = np.quantile(paths, alpha, axis=0)
                upper = np.quantile(paths, 1.0 - alpha, axis=0)
                # Build historical segment based on history_win (e.g., 60d)
                import pandas as _pd  # local alias to avoid top deps noise
                try:
                    days_num = int("".join(ch for ch in history_win if ch.isdigit())) or 60
                except Exception:
                    days_num = 60
                end_dt = s.index[-1] if not s.empty else _pd.Timestamp.utcnow()
                start_dt = end_dt - _pd.Timedelta(days=days_num)
                hist = s[s.index >= start_dt]

                # Build forecast x-axis as dates aligned to interval
                def _freq_from_interval(iv: str) -> str:
                    iv = (iv or "1d").lower()
                    if iv.endswith("wk"):
                        return "W"
                    if iv.endswith("mo"):
                        return "M"
                    return "D"

                freq = _freq_from_interval(interval)
                fc_dates = _pd.date_range(start=end_dt, periods=steps + 1, freq=freq)
                mean_series = np.concatenate([[S0], mean_path])
                lower_series = np.concatenate([[S0], lower])
                upper_series = np.concatenate([[S0], upper])
                if go is not None:
                    _template = "plotly_dark"
                    figure = go.Figure()
                    # History
                    figure.add_trace(go.Scatter(x=hist.index, y=hist.values, line=dict(color="#8892a6"), name="History", showlegend=True))
                    # Forecast band and mean
                    figure.add_trace(go.Scatter(x=fc_dates, y=upper_series, line=dict(color="#aac"), name="Upper", showlegend=False))
                    figure.add_trace(
                        go.Scatter(x=fc_dates, y=lower_series, line=dict(color="#aac"), fill="tonexty", name="Lower", showlegend=False)
                    )
                    figure.add_trace(
                        go.Scatter(x=fc_dates, y=mean_series, line=dict(color="#06c"), name="Estimated path", showlegend=True)
                    )
                    figure.update_layout(
                        title=f"Estimated movement for {t_disp} over {steps} steps ({time_h}) with history {history_win}",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template=_template,
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#0f172a",
                        height=360,
                        margin=dict(l=40, r=10, t=40, b=40),
                    )
                else:
                    figure = {
                        "data": [
                            {"x": hist.index.tolist(), "y": hist.values.tolist(), "type": "scatter", "name": "History", "line": {"color": "#8892a6"}},
                            {"x": fc_dates.tolist(), "y": upper_series.tolist(), "type": "scatter", "name": "Upper", "line": {"color": "#aac"}},
                            {"x": fc_dates.tolist(), "y": lower_series.tolist(), "type": "scatter", "name": "Lower", "fill": "tonexty", "line": {"color": "#aac"}},
                            {"x": fc_dates.tolist(), "y": mean_series.tolist(), "type": "scatter", "name": "Estimated path", "line": {"color": "#06c"}},
                        ],
                        "layout": {
                            "title": f"Estimated movement for {t_disp} over {steps} steps ({time_h}) with history {history_win}",
                            "xaxis": {"title": "Date"},
                            "yaxis": {"title": "Price"},
                            "paper_bgcolor": "#0f172a",
                            "plot_bgcolor": "#0f172a",
                            "height": 360,
                            "margin": {"l": 40, "r": 10, "t": 40, "b": 40},
                        },
                    }
            return data, cols, msg, figure
        except Exception as e:
            from dash import no_update
            # Log error and return
            import traceback, datetime
            logs_dir = Path(__file__).resolve().parents[2] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "error.log"
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{ts}] Predict callback error: {e!r}\n")
                f.write(traceback.format_exc() + "\n")
            return no_update, no_update, f"Error: {e}", no_update

    return app


if __name__ == "__main__":
    try:
        app = create_app()
        # Expose the server for deployment if needed
        server = app.server  # noqa: F841
        # Dash >=3 uses app.run instead of app.run_server
        app.run(host="0.0.0.0", port=8050, debug=False)
    except Exception as e:
        # Log any top-level startup/runtime error
        import traceback, datetime

        logs_dir = Path(__file__).resolve().parents[2] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "error.log"
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] Fatal error in app: {e!r}\n")
            f.write(traceback.format_exc() + "\n")
        raise
