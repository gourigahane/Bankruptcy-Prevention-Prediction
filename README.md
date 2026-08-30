# Bankruptcy Prevention Prediction

I built this as a small end-to-end ML project: take a classic bankruptcy
dataset, explore it, train a model, and actually ship something you can
click around in — instead of leaving the results buried in a notebook.

The dataset covers 250 companies, each scored on six qualitative risk
factors (rated Low / Medium / High), with a label saying whether the
company went bankrupt. I used that to train a Logistic Regression model
and wrapped it in a simple Streamlit app so anyone can plug in numbers
and get a prediction back in seconds.

## What it looks at

| Factor | What it means |
|---|---|
| Industrial Risk | How risky the industry itself is |
| Management Risk | How stable/capable the leadership is |
| Financial Flexibility | Room to maneuver financially if things get tight |
| Credibility | How trustworthy the company looks to outsiders |
| Competitiveness | How well it stacks up against competitors |
| Operating Risk | Risk baked into day-to-day operations |

You rate each one as 0 (low), 0.5 (medium), or 1 (high), and the app
predicts whether the company is likely to end up bankrupt or not — along
with how confident the model is.

## How it's put together

- `train_model.py` — loads the data, splits it into train/test sets
  (carefully grouping identical rows so duplicates don't leak between the
  two and inflate accuracy), and fits the Logistic Regression model.
- `Bankruptcy_Prevention.pkl` — the trained model, saved with pickle so it
  doesn't need to be retrained every time.
- `app.py` — a Streamlit front end that loads the model and turns it into
  something interactive.

## Running it yourself

```bash
pip install -r requirements.txt
python train_model.py --data Bankruptcy_prevention_Dataset.csv   # trains and saves the model
streamlit run app.py
```

## Putting it online

Easiest path is Streamlit Community Cloud — push this folder (pickle file
included) to a GitHub repo, point [share.streamlit.io](https://share.streamlit.io)
at it with `app.py` as the entry file, and it's live.

## Built with

Python, scikit-learn, pandas, and Streamlit.
