import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tkinter import filedialog, messagebox, ttk


CONFIG_FILENAME = "feature_config.json"

DEFAULT_FEATURES_CONFIG = [
    {"label": "Clicks", "key": "clicks", "min": 0, "example": "160"},
    {"label": "Dwell Time (seconds)", "key": "dwell_time", "min": 0, "example": "145"},
    {
        "label": "Keyword Relevance (0-1)",
        "key": "keyword_relevance",
        "min": 0,
        "max": 1,
        "example": "0.78",
    },
    {
        "label": "Page Load Time (seconds)",
        "key": "load_time",
        "min": 0,
        "strict_min": True,
        "example": "1.9",
    },
    {"label": "Bounce Rate (%)", "key": "bounce_rate", "min": 0, "max": 100, "example": "37"},
]

DEFAULT_CONFIG = {
    "label_column": "label",
    "url_column": "page_url",
    "features": DEFAULT_FEATURES_CONFIG,
}

INTENTIONS = {
    "Triage Content Backlog": (
        "Use this when you have many pages and need to decide which pages"
        " deserve optimization effort first."
    ),
    "Gate New Page Before Publish": (
        "Use this before release to decide if the page should go live now"
        " or be improved first."
    ),
    "Diagnose Traffic Decline": (
        "Use this when performance dropped and you need practical next-step"
        " checks."
    ),
}

POSITIVE_LABELS = {"1", "high", "true", "yes", "good", "positive"}
NEGATIVE_LABELS = {"0", "low", "false", "no", "bad", "negative"}


@dataclass
class FeatureSpec:
    label: str
    key: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    strict_min: bool = False
    example: str = ""


def _to_float_or_none(value) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def ensure_default_config(config_path: str) -> None:
    if os.path.exists(config_path):
        return
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=2)


def load_runtime_config(config_path: str) -> Tuple[List[FeatureSpec], str, str]:
    ensure_default_config(config_path)

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    label_column = str(raw.get("label_column", "label")).strip() or "label"
    url_column = str(raw.get("url_column", "page_url")).strip() or "page_url"

    features_raw = raw.get("features", [])
    if not isinstance(features_raw, list) or not features_raw:
        features_raw = DEFAULT_FEATURES_CONFIG

    feature_specs: List[FeatureSpec] = []
    for item in features_raw:
        if not isinstance(item, dict):
            continue

        key = str(item.get("key", "")).strip()
        label = str(item.get("label", "")).strip()
        if not key or not label:
            continue

        spec = FeatureSpec(
            label=label,
            key=key,
            min_value=_to_float_or_none(item.get("min")),
            max_value=_to_float_or_none(item.get("max")),
            strict_min=bool(item.get("strict_min", False)),
            example=str(item.get("example", "")).strip(),
        )
        feature_specs.append(spec)

    if not feature_specs:
        raise ValueError("Feature config is invalid: define at least one feature in feature_config.json.")

    return feature_specs, label_column, url_column


def build_sample_training_data(feature_keys: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    expected = ["clicks", "dwell_time", "keyword_relevance", "load_time", "bounce_rate"]
    if feature_keys != expected:
        return None

    # [Clicks, Dwell Time, Keyword Relevance, Page Load Time, Bounce Rate]
    X = np.array(
        [
            [220, 210, 0.94, 1.2, 16],
            [180, 165, 0.88, 1.5, 24],
            [145, 140, 0.82, 1.7, 32],
            [95, 80, 0.61, 2.4, 58],
            [70, 55, 0.52, 3.1, 71],
            [210, 190, 0.91, 1.3, 19],
            [130, 110, 0.68, 2.0, 46],
            [165, 150, 0.79, 1.8, 35],
            [85, 70, 0.57, 2.7, 62],
            [240, 220, 0.95, 1.1, 14],
            [155, 135, 0.74, 1.9, 39],
            [60, 42, 0.45, 3.4, 77],
            [195, 176, 0.86, 1.4, 27],
            [118, 95, 0.63, 2.3, 53],
            [175, 160, 0.84, 1.6, 30],
        ],
        dtype=float,
    )
    y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=int)
    return X, y


def parse_binary_label(raw_value) -> int:
    text = str(raw_value).strip().lower()
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0

    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported label value '{raw_value}'. Use HIGH/LOW or 1/0.") from exc

    return 1 if numeric > 0 else 0


def validate_feature_vector(values: np.ndarray, feature_specs: List[FeatureSpec], context: str) -> None:
    for index, spec in enumerate(feature_specs):
        value = float(values[index])

        if spec.min_value is not None:
            if spec.strict_min and value <= spec.min_value:
                raise ValueError(f"{context}: '{spec.key}' must be greater than {spec.min_value}.")
            if not spec.strict_min and value < spec.min_value:
                raise ValueError(f"{context}: '{spec.key}' must be at least {spec.min_value}.")

        if spec.max_value is not None and value > spec.max_value:
            raise ValueError(f"{context}: '{spec.key}' must be at most {spec.max_value}.")


def read_training_csv(
    path: str,
    feature_specs: List[FeatureSpec],
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        required = [spec.key for spec in feature_specs] + [label_column]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Training CSV is missing required columns: {', '.join(missing)}")

        X_rows: List[List[float]] = []
        y_rows: List[int] = []

        for line_number, row in enumerate(reader, start=2):
            row_values: List[float] = []
            for spec in feature_specs:
                raw_value = (row.get(spec.key) or "").strip()
                if raw_value == "":
                    raise ValueError(f"Row {line_number}: '{spec.key}' is empty.")
                try:
                    row_values.append(float(raw_value))
                except ValueError as exc:
                    raise ValueError(
                        f"Row {line_number}: '{spec.key}' must be numeric, got '{raw_value}'."
                    ) from exc

            vector = np.array(row_values, dtype=float)
            validate_feature_vector(vector, feature_specs, f"Row {line_number}")

            raw_label = row.get(label_column)
            if raw_label is None or str(raw_label).strip() == "":
                raise ValueError(f"Row {line_number}: '{label_column}' is empty.")

            X_rows.append(row_values)
            y_rows.append(parse_binary_label(raw_label))

    if not X_rows:
        raise ValueError("Training CSV contains no data rows.")

    y_array = np.array(y_rows, dtype=int)
    if len(set(y_array.tolist())) < 2:
        raise ValueError("Training CSV must contain both classes (HIGH and LOW).")

    return np.array(X_rows, dtype=float), y_array


def read_batch_csv(
    path: str,
    feature_specs: List[FeatureSpec],
    url_column: str,
) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        required = [spec.key for spec in feature_specs]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Batch CSV is missing required columns: {', '.join(missing)}")

        records: List[Dict[str, object]] = []
        for line_number, row in enumerate(reader, start=2):
            row_values: List[float] = []
            raw_map: Dict[str, float] = {}

            for spec in feature_specs:
                raw_value = (row.get(spec.key) or "").strip()
                if raw_value == "":
                    raise ValueError(f"Row {line_number}: '{spec.key}' is empty.")
                try:
                    numeric_value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Row {line_number}: '{spec.key}' must be numeric, got '{raw_value}'."
                    ) from exc
                row_values.append(numeric_value)
                raw_map[spec.key] = numeric_value

            vector = np.array(row_values, dtype=float)
            validate_feature_vector(vector, feature_specs, f"Row {line_number}")

            page_url = (
                (row.get(url_column) or "").strip()
                or (row.get("page_url") or "").strip()
                or (row.get("url") or "").strip()
                or (row.get("page") or "").strip()
                or f"Row {line_number - 1}"
            )

            records.append(
                {
                    "row_number": line_number,
                    "page_url": page_url,
                    "values": vector,
                    "raw_map": raw_map,
                }
            )

    if not records:
        raise ValueError("Batch CSV contains no data rows.")

    return records


def margin_to_confidence(margin: float) -> float:
    return float(1.0 / (1.0 + np.exp(-abs(margin))))


def train_model_with_metrics(X: np.ndarray, y: np.ndarray):
    classes = set(y.tolist())
    if classes != {0, 1}:
        raise ValueError("Training data must contain exactly two classes encoded as 0/1.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel="linear")
    model.fit(X_scaled, y)

    class_counts = Counter(y.tolist())
    total = len(y)
    majority_baseline = max(class_counts.values()) / total

    metrics = {
        "samples": total,
        "low_count": class_counts.get(0, 0),
        "high_count": class_counts.get(1, 0),
        "majority_baseline": majority_baseline,
        "cv_splits": 0,
        "cv_accuracy": None,
        "low_precision": None,
        "low_recall": None,
        "high_precision": None,
        "high_recall": None,
        "warning": "",
    }

    min_class_size = min(class_counts.values())
    if min_class_size >= 2:
        cv_splits = min(5, min_class_size)
        cv_model = SVC(kernel="linear")
        splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        predictions = cross_val_predict(cv_model, X_scaled, y, cv=splitter)

        accuracy = float(accuracy_score(y, predictions))
        precision, recall, _, _ = precision_recall_fscore_support(
            y, predictions, labels=[0, 1], zero_division=0
        )

        metrics.update(
            {
                "cv_splits": cv_splits,
                "cv_accuracy": accuracy,
                "low_precision": float(precision[0]),
                "low_recall": float(recall[0]),
                "high_precision": float(precision[1]),
                "high_recall": float(recall[1]),
            }
        )

        if accuracy <= majority_baseline + 0.03:
            metrics["warning"] = (
                "CV accuracy is close to majority-class baseline; precision/recall should drive trust."
            )
    else:
        metrics["warning"] = (
            "Class counts are too small for stable cross-validation. Add more labeled rows."
        )

    return scaler, model, metrics


class IntentDrivenSVMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intent-Driven SVM Ranking Assistant")
        self.root.geometry("1060x760")
        self.root.minsize(980, 700)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(base_dir, CONFIG_FILENAME)
        self.feature_specs, self.label_column, self.url_column = load_runtime_config(self.config_path)

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[SVC] = None
        self.coefficients: Optional[np.ndarray] = None
        self.model_metrics: Optional[Dict[str, object]] = None
        self.training_source = "Not trained"

        self.intention_var = tk.StringVar(value=list(INTENTIONS.keys())[0])
        self.feature_vars = {spec.key: tk.StringVar() for spec in self.feature_specs}

        self.model_status_var = tk.StringVar(value="Model not trained yet.")
        self.confidence_value_var = tk.DoubleVar(value=0.0)
        self.confidence_text_var = tk.StringVar(value="Confidence: n/a")
        self.batch_file_var = tk.StringVar(value="No batch file loaded.")
        self.batch_summary_var = tk.StringVar(value="Load a batch CSV to start.")

        self.batch_records: List[Dict[str, object]] = []
        self.batch_results: List[Dict[str, object]] = []
        self.history_records: List[Dict[str, object]] = []
        self.history_dirty = False

        self._build_ui()
        self._fill_example_values()
        self._bootstrap_model()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _required_feature_columns(self) -> str:
        return ", ".join(spec.key for spec in self.feature_specs)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            outer,
            text="Practical Search Page Ranking Assistant",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor=tk.W)

        ttk.Label(
            outer,
            text=(
                "Load real training data, evaluate single pages, then run batch audits"
                " to prioritize work by SVM decision score."
            ),
            font=("Segoe UI", 10),
        ).pack(anchor=tk.W, pady=(2, 10))

        model_frame = ttk.LabelFrame(outer, text="Model & Data Setup", padding=10)
        model_frame.pack(fill=tk.X)

        ttk.Button(model_frame, text="Load Training CSV", command=self.load_training_csv).grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Button(
            model_frame, text="Use Built-in Sample Data", command=self.use_sample_training_data
        ).grid(row=0, column=1, sticky=tk.W, padx=(8, 0))

        ttk.Label(
            model_frame,
            text=(
                f"Feature config: {self.config_path}"
                f" | Label column: {self.label_column}"
                f" | Batch URL column: {self.url_column}"
            ),
            wraplength=980,
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(8, 0))

        ttk.Label(
            model_frame,
            text=f"Required feature columns: {self._required_feature_columns()}",
            wraplength=980,
            font=("Segoe UI", 9),
        ).grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(2, 0))

        ttk.Label(
            model_frame,
            textvariable=self.model_status_var,
            justify=tk.LEFT,
            wraplength=980,
            font=("Segoe UI", 9, "italic"),
        ).grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(8, 0))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        single_tab = ttk.Frame(notebook, padding=10)
        batch_tab = ttk.Frame(notebook, padding=10)
        history_tab = ttk.Frame(notebook, padding=10)

        notebook.add(single_tab, text="Single Page Evaluation")
        notebook.add(batch_tab, text="Batch Audit")
        notebook.add(history_tab, text="Session History")

        self._build_single_tab(single_tab)
        self._build_batch_tab(batch_tab)
        self._build_history_tab(history_tab)

    def _build_single_tab(self, parent):
        intent_frame = ttk.LabelFrame(parent, text="1) User Intention", padding=10)
        intent_frame.pack(fill=tk.X)

        ttk.Label(intent_frame, text="What do you want this model to help with?").grid(
            row=0, column=0, sticky=tk.W
        )
        self.intent_combo = ttk.Combobox(
            intent_frame,
            values=list(INTENTIONS.keys()),
            state="readonly",
            textvariable=self.intention_var,
            width=34,
        )
        self.intent_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        self.intent_combo.bind("<<ComboboxSelected>>", self._on_intention_changed)

        self.intent_desc = ttk.Label(
            intent_frame,
            text=INTENTIONS[self.intention_var.get()],
            wraplength=920,
        )
        self.intent_desc.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

        signals_frame = ttk.LabelFrame(parent, text="2) Page Signals", padding=10)
        signals_frame.pack(fill=tk.X, pady=(10, 0))

        for row, spec in enumerate(self.feature_specs):
            ttk.Label(signals_frame, text=spec.label).grid(row=row, column=0, sticky=tk.W, pady=4)
            ttk.Entry(signals_frame, textvariable=self.feature_vars[spec.key], width=20).grid(
                row=row, column=1, sticky=tk.W, padx=(10, 0), pady=4
            )

        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(controls, text="Evaluate", command=self.evaluate_single_page).pack(side=tk.LEFT)
        ttk.Button(controls, text="Load Example", command=self._fill_example_values).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(controls, text="Reset", command=self._reset_single_inputs).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        confidence_frame = ttk.LabelFrame(parent, text="3) Confidence & Margin", padding=10)
        confidence_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(confidence_frame, textvariable=self.confidence_text_var).pack(anchor=tk.W)
        ttk.Progressbar(
            confidence_frame,
            maximum=100,
            variable=self.confidence_value_var,
            length=760,
        ).pack(fill=tk.X, pady=(6, 0))

        output_frame = ttk.LabelFrame(parent, text="4) Result & Action Plan", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=12, font=("Consolas", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self._set_output("Enter page signals and click Evaluate.")

    def _build_batch_tab(self, parent):
        ttk.Label(
            parent,
            text=(
                "Load a CSV with one row per page and columns for all configured features."
                " Results are sorted by decision margin (lowest first = highest priority)."
            ),
            wraplength=940,
        ).pack(anchor=tk.W)

        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(actions, text="Load Batch CSV", command=self.load_batch_csv).pack(side=tk.LEFT)
        ttk.Button(actions, text="Run Audit", command=self.run_batch_audit).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="Export Results CSV", command=self.export_batch_results).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        ttk.Label(parent, textvariable=self.batch_file_var, wraplength=940).pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(
            parent,
            text=f"Required feature columns: {self._required_feature_columns()}",
            wraplength=940,
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(parent, textvariable=self.batch_summary_var, wraplength=940).pack(
            anchor=tk.W, pady=(4, 0)
        )

        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        columns = ("url", "prediction", "confidence", "margin", "top_signal")
        self.batch_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)

        self.batch_tree.heading("url", text="Page URL")
        self.batch_tree.heading("prediction", text="Predicted Label")
        self.batch_tree.heading("confidence", text="Confidence")
        self.batch_tree.heading("margin", text="Decision Margin")
        self.batch_tree.heading("top_signal", text="Top Signal")

        self.batch_tree.column("url", width=330, anchor=tk.W)
        self.batch_tree.column("prediction", width=120, anchor=tk.CENTER)
        self.batch_tree.column("confidence", width=110, anchor=tk.CENTER)
        self.batch_tree.column("margin", width=120, anchor=tk.CENTER)
        self.batch_tree.column("top_signal", width=230, anchor=tk.W)

        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.batch_tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.batch_tree.xview)
        self.batch_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.batch_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def _build_history_tab(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Export Session Log CSV", command=self.export_session_history).pack(
            side=tk.LEFT
        )
        ttk.Button(controls, text="Clear Session Log", command=self.clear_session_history).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        columns = ("time", "mode", "page", "prediction", "confidence", "margin", "top_signal")
        self.history_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)

        self.history_tree.heading("time", text="Timestamp")
        self.history_tree.heading("mode", text="Mode")
        self.history_tree.heading("page", text="Page")
        self.history_tree.heading("prediction", text="Prediction")
        self.history_tree.heading("confidence", text="Confidence")
        self.history_tree.heading("margin", text="Margin")
        self.history_tree.heading("top_signal", text="Top Signal")

        self.history_tree.column("time", width=145, anchor=tk.W)
        self.history_tree.column("mode", width=70, anchor=tk.CENTER)
        self.history_tree.column("page", width=280, anchor=tk.W)
        self.history_tree.column("prediction", width=90, anchor=tk.CENTER)
        self.history_tree.column("confidence", width=90, anchor=tk.CENTER)
        self.history_tree.column("margin", width=90, anchor=tk.CENTER)
        self.history_tree.column("top_signal", width=230, anchor=tk.W)

        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.history_tree.xview)
        self.history_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.history_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def _bootstrap_model(self):
        sample_data = build_sample_training_data([spec.key for spec in self.feature_specs])
        if sample_data is None:
            self.model_status_var.set(
                "No compatible built-in sample dataset for this feature config. Load a training CSV."
            )
            return

        X, y = sample_data
        self._apply_training_data(X, y, "Built-in sample data")

    def _format_model_status(self) -> str:
        if not self.model_metrics:
            return "Model not trained yet."

        metrics = self.model_metrics
        lines = [
            f"Training source: {self.training_source}",
            (
                f"Samples: {metrics['samples']}"
                f" | Class balance LOW/HIGH: {metrics['low_count']}/{metrics['high_count']}"
                f" | Majority baseline: {metrics['majority_baseline']:.2f}"
            ),
        ]

        if metrics["cv_accuracy"] is not None:
            lines.append(
                f"{metrics['cv_splits']}-fold CV accuracy: {metrics['cv_accuracy']:.2f}"
                f" | LOW precision/recall: {metrics['low_precision']:.2f}/{metrics['low_recall']:.2f}"
                f" | HIGH precision/recall: {metrics['high_precision']:.2f}/{metrics['high_recall']:.2f}"
            )

        warning = metrics.get("warning") or ""
        if warning:
            lines.append(f"Warning: {warning}")

        return "\n".join(lines)

    def _apply_training_data(self, X: np.ndarray, y: np.ndarray, source: str):
        scaler, model, metrics = train_model_with_metrics(X, y)
        self.scaler = scaler
        self.model = model
        self.coefficients = model.coef_[0]
        self.model_metrics = metrics
        self.training_source = source
        self.model_status_var.set(self._format_model_status())

    def _set_output(self, text: str):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.config(state=tk.DISABLED)

    def _on_intention_changed(self, _event=None):
        self.intent_desc.config(text=INTENTIONS[self.intention_var.get()])

    def _fill_example_values(self):
        for spec in self.feature_specs:
            self.feature_vars[spec.key].set(spec.example)

    def _reset_single_inputs(self):
        for spec in self.feature_specs:
            self.feature_vars[spec.key].set("")
        self.confidence_value_var.set(0.0)
        self.confidence_text_var.set("Confidence: n/a")
        self._set_output("Enter page signals and click Evaluate.")

    def _read_single_features(self) -> Tuple[np.ndarray, Dict[str, float]]:
        values: List[float] = []
        raw_map: Dict[str, float] = {}

        for spec in self.feature_specs:
            raw_value = self.feature_vars[spec.key].get().strip()
            if raw_value == "":
                raise ValueError(f"{spec.label} is required.")
            try:
                numeric = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"{spec.label} must be numeric.") from exc

            values.append(numeric)
            raw_map[spec.key] = numeric

        vector = np.array(values, dtype=float)
        validate_feature_vector(vector, self.feature_specs, "Input")
        return vector, raw_map

    def _is_model_ready(self) -> bool:
        return self.scaler is not None and self.model is not None and self.coefficients is not None

    def _feature_contributions(self, scaled_values: np.ndarray) -> List[Tuple[str, float]]:
        contributions = self.coefficients * scaled_values
        labeled = [(self.feature_specs[i].label, float(contributions[i])) for i in range(len(contributions))]
        labeled.sort(key=lambda item: abs(item[1]), reverse=True)
        return labeled

    def _top_signal_text(self, contributions: List[Tuple[str, float]]) -> str:
        if not contributions:
            return "n/a"
        name, impact = contributions[0]
        return f"{name} ({'helping' if impact >= 0 else 'hurting'})"

    def _find_signal_by_key_token(self, raw_values: Dict[str, float], tokens: List[str]) -> Optional[float]:
        for spec in self.feature_specs:
            key = spec.key.lower()
            if any(token in key for token in tokens):
                return raw_values.get(spec.key)
        return None

    def _intention_actions(
        self,
        intention: str,
        predicted_high: bool,
        confidence: float,
        raw_values: Dict[str, float],
    ) -> str:
        bounce = self._find_signal_by_key_token(raw_values, ["bounce"])
        load_time = self._find_signal_by_key_token(raw_values, ["load", "latency", "lcp"])
        relevance = self._find_signal_by_key_token(raw_values, ["relevance", "keyword"])
        dwell = self._find_signal_by_key_token(raw_values, ["dwell", "time_on_page"])

        if intention == "Gate New Page Before Publish":
            if predicted_high and confidence >= 0.75:
                return (
                    "Decision: Publish now.\n"
                    "Reason: The page clears a strict go-live confidence threshold.\n"
                    "Action: Launch and monitor post-publish behavior for 7 days."
                )
            return (
                "Decision: Hold release.\n"
                "Reason: Confidence is not strong enough for go-live quality.\n"
                "Action: Improve weak signals and re-evaluate before publishing."
            )

        if intention == "Diagnose Traffic Decline":
            checks: List[str] = []
            if bounce is not None and bounce > 55:
                checks.append("Bounce-like signal is high; revisit intent match and CTA placement.")
            if load_time is not None and load_time > 2.4:
                checks.append("Load-performance signal is weak; prioritize speed fixes.")
            if relevance is not None and relevance < 0.65:
                checks.append("Relevance signal is low; rewrite title, H1, and intro alignment.")
            if dwell is not None and dwell < 90:
                checks.append("Engagement-time signal is weak; strengthen above-the-fold clarity.")
            if not checks:
                checks.append("No obvious red flags; inspect SERP competition and freshness changes.")
            return "Decision: Run a decline investigation.\nAction checks:\n" + "\n".join(
                f"- {item}" for item in checks
            )

        if predicted_high and confidence >= 0.6:
            return (
                "Decision: Keep this page in maintain mode.\n"
                "Action: Focus optimization time on weaker pages first."
            )

        return (
            "Decision: Prioritize this page in the optimization backlog.\n"
            "Action: Improve the most negative signals before promotion."
        )

    def _set_confidence_display(self, confidence: float, margin: float):
        abs_margin = abs(margin)
        if abs_margin < 0.25:
            boundary_note = "near boundary"
        elif abs_margin < 0.75:
            boundary_note = "moderate separation"
        else:
            boundary_note = "clear separation"

        self.confidence_value_var.set(round(confidence * 100, 2))
        self.confidence_text_var.set(
            f"Confidence: {confidence:.2f} | Margin: {margin:+.2f} ({boundary_note})"
        )

    def _append_history(self, entry: Dict[str, object]):
        self.history_records.append(entry)
        self.history_tree.insert(
            "",
            tk.END,
            values=(
                entry["timestamp"],
                entry["mode"],
                entry["page"],
                entry["prediction"],
                f"{entry['confidence']:.2f}",
                f"{entry['margin']:+.2f}",
                entry["top_signal"],
            ),
        )
        self.history_dirty = True

    def evaluate_single_page(self):
        if not self._is_model_ready():
            self._set_output("Model is not trained. Load a training CSV first.")
            return

        try:
            raw_vector, raw_map = self._read_single_features()
        except ValueError as error:
            self._set_output(f"Input error: {error}")
            return

        scaled = self.scaler.transform([raw_vector])
        prediction = int(self.model.predict(scaled)[0])
        margin = float(self.model.decision_function(scaled)[0])
        confidence = margin_to_confidence(margin)
        predicted_high = prediction == 1

        contributions = self._feature_contributions(scaled[0])
        top_signal = self._top_signal_text(contributions)
        top_lines = [
            f"- {name}: {impact:+.2f} ({'supports ranking' if impact >= 0 else 'hurts ranking'})"
            for name, impact in contributions[:3]
        ]

        intention = self.intention_var.get()
        class_label = "HIGH ranking potential" if predicted_high else "LOW ranking potential"
        action_plan = self._intention_actions(intention, predicted_high, confidence, raw_map)

        self._set_confidence_display(confidence, margin)
        result = (
            f"Intention: {intention}\n"
            f"Prediction: {class_label}\n"
            f"Confidence score: {confidence:.2f}\n"
            f"Decision margin: {margin:+.2f}\n"
            f"\nTop feature pressure:\n{chr(10).join(top_lines)}\n"
            f"\nDominant signal: {top_signal}\n"
            f"\n{action_plan}\n"
        )
        self._set_output(result)

        self._append_history(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "single",
                "page": "manual-input",
                "prediction": "HIGH" if predicted_high else "LOW",
                "confidence": confidence,
                "margin": margin,
                "top_signal": top_signal,
            }
        )

    def load_training_csv(self):
        path = filedialog.askopenfilename(
            title="Select training CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return

        try:
            X, y = read_training_csv(path, self.feature_specs, self.label_column)
            self._apply_training_data(X, y, os.path.basename(path))
        except Exception as error:
            messagebox.showerror("Training Data Error", str(error))
            return

        messagebox.showinfo("Training Complete", f"Model retrained from:\n{path}")

    def use_sample_training_data(self):
        sample_data = build_sample_training_data([spec.key for spec in self.feature_specs])
        if sample_data is None:
            messagebox.showwarning(
                "Sample Data Unavailable",
                "Built-in sample data only works with default feature keys.\n"
                "Load a real training CSV for this custom feature setup.",
            )
            return

        X, y = sample_data
        self._apply_training_data(X, y, "Built-in sample data")
        messagebox.showinfo("Training Complete", "Model reset to built-in sample data.")

    def load_batch_csv(self):
        path = filedialog.askopenfilename(
            title="Select batch input CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return

        try:
            self.batch_records = read_batch_csv(path, self.feature_specs, self.url_column)
        except Exception as error:
            messagebox.showerror("Batch CSV Error", str(error))
            return

        self.batch_results = []
        for row_id in self.batch_tree.get_children():
            self.batch_tree.delete(row_id)

        self.batch_file_var.set(f"Loaded: {path}")
        self.batch_summary_var.set(f"Loaded {len(self.batch_records)} pages. Click 'Run Audit'.")

    def run_batch_audit(self):
        if not self._is_model_ready():
            messagebox.showwarning("Model Not Ready", "Load training data before running batch audit.")
            return

        if not self.batch_records:
            messagebox.showwarning("No Batch Data", "Load a batch CSV first.")
            return

        matrix = np.array([record["values"] for record in self.batch_records], dtype=float)
        scaled = self.scaler.transform(matrix)
        predictions = self.model.predict(scaled)
        margins = self.model.decision_function(scaled)

        results: List[Dict[str, object]] = []
        for idx, record in enumerate(self.batch_records):
            prediction = int(predictions[idx])
            margin = float(margins[idx])
            confidence = margin_to_confidence(margin)
            contributions = self._feature_contributions(scaled[idx])
            top_signal = self._top_signal_text(contributions)
            label = "HIGH" if prediction == 1 else "LOW"

            result = {
                "page_url": record["page_url"],
                "prediction": label,
                "confidence": confidence,
                "margin": margin,
                "top_signal": top_signal,
            }
            results.append(result)

            self._append_history(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "batch",
                    "page": record["page_url"],
                    "prediction": label,
                    "confidence": confidence,
                    "margin": margin,
                    "top_signal": top_signal,
                }
            )

        results.sort(key=lambda row: row["margin"])
        self.batch_results = results
        self._render_batch_results()

        low_count = sum(1 for row in results if row["prediction"] == "LOW")
        self.batch_summary_var.set(
            f"Audited {len(results)} pages | LOW: {low_count} | HIGH: {len(results) - low_count}"
            " | Sorted by decision margin (lowest first)."
        )

    def _render_batch_results(self):
        for row_id in self.batch_tree.get_children():
            self.batch_tree.delete(row_id)

        for result in self.batch_results:
            self.batch_tree.insert(
                "",
                tk.END,
                values=(
                    result["page_url"],
                    result["prediction"],
                    f"{result['confidence']:.2f}",
                    f"{result['margin']:+.2f}",
                    result["top_signal"],
                ),
            )

    def export_batch_results(self):
        if not self.batch_results:
            messagebox.showwarning("No Results", "Run a batch audit before exporting.")
            return

        default_name = f"batch_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            title="Export batch results",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "page_url",
                    "predicted_label",
                    "confidence_score",
                    "decision_margin",
                    "top_contributing_signal",
                ]
            )
            for result in self.batch_results:
                writer.writerow(
                    [
                        result["page_url"],
                        result["prediction"],
                        f"{result['confidence']:.4f}",
                        f"{result['margin']:.4f}",
                        result["top_signal"],
                    ]
                )

        messagebox.showinfo("Export Complete", f"Batch results saved to:\n{path}")

    def export_session_history(self) -> bool:
        if not self.history_records:
            messagebox.showwarning("No History", "No session records to export.")
            return False

        default_name = f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            title="Export session history",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return False

        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "mode", "page", "prediction", "confidence", "margin", "top_signal"])
            for entry in self.history_records:
                writer.writerow(
                    [
                        entry["timestamp"],
                        entry["mode"],
                        entry["page"],
                        entry["prediction"],
                        f"{entry['confidence']:.4f}",
                        f"{entry['margin']:.4f}",
                        entry["top_signal"],
                    ]
                )

        self.history_dirty = False
        messagebox.showinfo("Export Complete", f"Session history saved to:\n{path}")
        return True

    def clear_session_history(self):
        if not self.history_records:
            return

        confirmed = messagebox.askyesno("Clear Session Log", "Delete all session history rows?")
        if not confirmed:
            return

        self.history_records = []
        self.history_dirty = False
        for row_id in self.history_tree.get_children():
            self.history_tree.delete(row_id)

    def _on_close(self):
        if self.history_records and self.history_dirty:
            decision = messagebox.askyesnocancel(
                "Export Session Log",
                "You have unsaved session history. Export it before exit?",
            )
            if decision is None:
                return
            if decision:
                exported = self.export_session_history()
                if not exported:
                    return
        self.root.destroy()


def main():
    root = tk.Tk()
    IntentDrivenSVMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
