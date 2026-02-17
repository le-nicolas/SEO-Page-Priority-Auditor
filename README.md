# Intent-Driven SVM Ranking Assistant

A one-file practical desktop tool for making page-ranking decisions with an SVM model.

This is not a toy chart demo. It is designed for day-to-day content and SEO operations where someone needs to decide:

- Which pages to optimize first
- Whether a new page is ready to publish
- What to check first when traffic drops

![Practical workflow](assets/practical-workflow.png)

## What It Does

- Uses a linear SVM to classify page ranking potential (`HIGH` or `LOW`)
- Uses a GUI intention selector to change how results are interpreted
- Gives actionable guidance based on the selected intention
- Shows model health (cross-validation accuracy) directly in the UI

## Single-Code-File Structure

- App logic is in one file: `main.py`

## Run Locally

```bash
python -m pip install numpy scikit-learn
python main.py
```

## Input Signals

The app evaluates these practical page signals:

- Clicks
- Dwell Time (seconds)
- Keyword Relevance (0-1)
- Page Load Time (seconds)
- Bounce Rate (%)

## Why This Is Practical

- Creates a repeatable decision workflow instead of gut-feel calls
- Makes prioritization explicit for editors, SEO analysts, and content leads
- Converts model output into next-step operational actions

## Typical Use

1. Select intention (triage, pre-publish gate, or decline diagnosis)
2. Enter page signals from analytics and page-speed tooling
3. Run evaluation
4. Act on the recommendation immediately
