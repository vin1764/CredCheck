# Deploying CredCheck on Streamlit (free public URL)

The app is `app.py`; it imports the model from `credcheck_model.py`. The model is trained
once on startup and cached, so there's no model file to ship.

## Run locally first
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. The first load trains the model (a few seconds); after that
it's instant (cached via `@st.cache_resource`).

## Put it online with Streamlit Community Cloud (free, ~5 minutes)
1. Create a GitHub repo with **these files at the root**:
   ```
   app.py
   credcheck_model.py
   requirements.txt
   (optional) thinfile_proprietorships.csv, rich_complete.csv, ...   # for the batch tab demo
   ```
2. Push to GitHub.
3. Go to **share.streamlit.io** → sign in with GitHub → **Create app** → **Deploy a public app from a repo**.
4. Select your repo and branch; set **Main file path = `app.py`** → **Deploy**.
5. You get a public URL like `https://your-app.streamlit.app` that anyone can open. Done.

Community Cloud installs `requirements.txt` automatically. The first visit trains the model
(~5–10 s on the free tier, well within limits); subsequent visits reuse the cached model.

## Optional: skip startup training (pre-baked model)
If you want zero training at runtime, persist the model once and load it instead:
```python
# build_model_artifact.py  (run locally, commit credcheck_model.joblib)
import joblib
from credcheck_model import build_default_model
model, _ = build_default_model()
joblib.dump(model, "credcheck_model.joblib")
```
Then in `app.py` replace `get_model()` with `joblib.load("credcheck_model.joblib")` and add
`joblib` to `requirements.txt`. (Keep scikit-learn versions matched between build and deploy,
or the pickle may not load — which is why training-on-startup is the safer default.)

## Other hosting options
- **Hugging Face Spaces** (Streamlit SDK): same files, also free, often snappier cold starts.
- **Render / Railway / Fly.io**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Docker**: `pip install -r requirements.txt`, then the run command above.

## Note for your submission
The deployed model is trained on **synthetic** data, so label it a *prototype / demo*. It
shows the mechanism (adaptive scoring, conservative sizing under missing data); production
would retrain on real GST / Account Aggregator / bureau / anchor-ledger data and re-run the
same validation (segment AUC, the hide-a-source ablation) on real outcomes.
