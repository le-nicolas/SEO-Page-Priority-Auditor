import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURES = (
    ("Clicks", "clicks"),
    ("Dwell Time (seconds)", "dwell_time"),
    ("Keyword Relevance (0-1)", "keyword_relevance"),
    ("Page Load Time (seconds)", "load_time"),
    ("Bounce Rate (%)", "bounce_rate"),
)

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


def build_training_data():
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

    # 1 = likely to rank well, 0 = at risk of poor ranking
    y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=int)
    return X, y


class IntentDrivenSVMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intent-Driven SVM Ranking Assistant")
        self.root.geometry("840x640")
        self.root.minsize(760, 580)

        self.scaler, self.model, self.cv_accuracy = self._train_model()
        self.coefficients = self.model.coef_[0]

        self.intention_var = tk.StringVar(value=list(INTENTIONS.keys())[0])
        self.feature_vars = {key: tk.StringVar() for _, key in FEATURES}
        self.results_var = tk.StringVar(value="Enter page signals and click Evaluate.")

        self._build_ui()
        self._fill_example_values()

    def _train_model(self):
        X, y = build_training_data()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = SVC(kernel="linear")
        model.fit(X_scaled, y)

        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        return scaler, model, float(cv_scores.mean())

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            outer,
            text="Practical Search Page Ranking Assistant",
            font=("Segoe UI", 15, "bold"),
        )
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            outer,
            text="Choose your intention, provide page signals, and get a direct action recommendation.",
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor=tk.W, pady=(2, 12))

        intent_frame = ttk.LabelFrame(outer, text="1) User Intention", padding=10)
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
            wraplength=700,
        )
        self.intent_desc.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

        signal_frame = ttk.LabelFrame(outer, text="2) Page Signals", padding=10)
        signal_frame.pack(fill=tk.X, pady=(12, 0))

        for row, (label, key) in enumerate(FEATURES):
            ttk.Label(signal_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
            ttk.Entry(
                signal_frame,
                textvariable=self.feature_vars[key],
                width=18,
            ).grid(row=row, column=1, sticky=tk.W, padx=(10, 0), pady=4)

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(12, 0))

        ttk.Button(buttons, text="Evaluate", command=self.evaluate).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Load Example", command=self._fill_example_values).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(buttons, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=(8, 0))

        model_label = ttk.Label(
            buttons,
            text=f"Model health (5-fold CV accuracy): {self.cv_accuracy:.2f}",
            font=("Segoe UI", 9, "italic"),
        )
        model_label.pack(side=tk.RIGHT)

        output = ttk.LabelFrame(outer, text="3) Result & Action Plan", padding=10)
        output.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.output_text = tk.Text(
            output,
            wrap=tk.WORD,
            height=16,
            font=("Consolas", 10),
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.insert("1.0", self.results_var.get())
        self.output_text.config(state=tk.DISABLED)

    def _on_intention_changed(self, _event=None):
        self.intent_desc.config(text=INTENTIONS[self.intention_var.get()])

    def _fill_example_values(self):
        defaults = {
            "clicks": "160",
            "dwell_time": "145",
            "keyword_relevance": "0.78",
            "load_time": "1.9",
            "bounce_rate": "37",
        }
        for key, value in defaults.items():
            self.feature_vars[key].set(value)

    def _reset(self):
        for _, key in FEATURES:
            self.feature_vars[key].set("")
        self._set_output("Enter page signals and click Evaluate.")

    def _set_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.config(state=tk.DISABLED)

    def _read_features(self):
        try:
            values = np.array(
                [
                    float(self.feature_vars["clicks"].get()),
                    float(self.feature_vars["dwell_time"].get()),
                    float(self.feature_vars["keyword_relevance"].get()),
                    float(self.feature_vars["load_time"].get()),
                    float(self.feature_vars["bounce_rate"].get()),
                ],
                dtype=float,
            )
        except ValueError as exc:
            raise ValueError("All fields must be numeric values.") from exc

        if values[0] < 0 or values[1] < 0:
            raise ValueError("Clicks and dwell time must be non-negative.")
        if not 0 <= values[2] <= 1:
            raise ValueError("Keyword relevance must be between 0 and 1.")
        if values[3] <= 0:
            raise ValueError("Page load time must be greater than 0.")
        if not 0 <= values[4] <= 100:
            raise ValueError("Bounce rate must be between 0 and 100.")

        return values

    def _explain_feature_pressure(self, scaled_values):
        contributions = self.coefficients * scaled_values
        labeled = list(zip([name for name, _ in FEATURES], contributions))
        labeled.sort(key=lambda item: abs(item[1]), reverse=True)

        top = labeled[:3]
        lines = []
        for name, impact in top:
            direction = "supports ranking" if impact >= 0 else "hurts ranking"
            lines.append(f"- {name}: {impact:+.2f} ({direction})")
        return "\n".join(lines)

    def _intention_actions(self, intention, predicted_high, confidence, raw_values):
        clicks, dwell_time, relevance, load_time, bounce = raw_values

        if intention == "Gate New Page Before Publish":
            if predicted_high and confidence >= 0.75:
                return (
                    "Decision: Publish now.\n"
                    "Reason: The page clears a strict go-live threshold.\n"
                    "Action: Launch and monitor bounce rate for 7 days."
                )
            return (
                "Decision: Hold release.\n"
                "Reason: Confidence is not strong enough for go-live quality.\n"
                "Action: Improve keyword relevance and load time, then reevaluate."
            )

        if intention == "Diagnose Traffic Decline":
            checks = []
            if bounce > 55:
                checks.append("Bounce rate is high; revisit intent match and CTA placement.")
            if load_time > 2.4:
                checks.append("Page load time is slow; prioritize performance fixes.")
            if relevance < 0.65:
                checks.append("Keyword relevance is low; rewrite title, H1, and intro.")
            if dwell_time < 90:
                checks.append("Dwell time is weak; improve information scent above the fold.")
            if not checks:
                checks.append("No obvious red flags; investigate SERP competition and freshness.")
            return "Decision: Run a decline investigation.\nAction checks:\n" + "\n".join(
                f"- {line}" for line in checks
            )

        if predicted_high and confidence >= 0.6:
            return (
                "Decision: Keep this page in maintain mode.\n"
                "Action: Focus optimization time on weaker pages first."
            )
        return (
            "Decision: Prioritize this page in the optimization backlog.\n"
            "Action: Improve relevance, speed, and bounce-rate levers before promotion."
        )

    def evaluate(self):
        try:
            raw_values = self._read_features()
        except ValueError as error:
            self._set_output(f"Input error: {error}")
            return

        scaled = self.scaler.transform([raw_values])
        prediction = int(self.model.predict(scaled)[0])
        margin = float(self.model.decision_function(scaled)[0])
        confidence = 1.0 / (1.0 + np.exp(-abs(margin)))
        predicted_high = bool(prediction == 1)

        intention = self.intention_var.get()
        classification = "HIGH ranking potential" if predicted_high else "LOW ranking potential"

        explanation = self._explain_feature_pressure(scaled[0])
        action_plan = self._intention_actions(
            intention=intention,
            predicted_high=predicted_high,
            confidence=confidence,
            raw_values=raw_values,
        )

        result = (
            f"Intention: {intention}\n"
            f"Prediction: {classification}\n"
            f"Confidence score: {confidence:.2f}\n"
            f"Decision margin: {margin:+.2f}\n"
            f"\nTop feature pressure:\n{explanation}\n"
            f"\n{action_plan}\n"
        )
        self._set_output(result)


def main():
    root = tk.Tk()
    app = IntentDrivenSVMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
