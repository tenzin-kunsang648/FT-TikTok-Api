# Managing Multiple Cohorts

## Running Multiple Batches Simultaneously

Each cohort is independent - can track multiple batches in parallel.

---

## Example Timeline

**Monday 12pm - Cohort A starts**
```bash
python3 tiktok_tracker.py init 50
# Creates: cohort_20251207_120000
```

**Tuesday 12pm - Cohort B starts (A still running)**
```bash
python3 tiktok_tracker.py init 50
# Creates: cohort_20251208_120000
```

Now managing 2 cohorts:
- A needs: hour_6, hour_24, hour_48
- B needs: hour_1, hour_6, hour_24, hour_48

---

## Commands for Specific Cohorts

```bash
# List all
python3 tiktok_tracker.py list

# Collect specific cohort
python3 tiktok_tracker.py collect 20251207_120000
python3 tiktok_tracker.py collect 20251208_120000

# Status of specific cohort
python3 tiktok_tracker.py status 20251207_120000

# Export specific cohort
python3 tiktok_tracker.py export 20251207_120000
```

---

## File Organization

```
data/
├── cohort_20251207_120000/    (Batch A - 50 videos)
│   ├── tracked_videos.json
│   ├── hour_0.json ... hour_48.json
│   └── final_dataset.csv
│
└── cohort_20251208_120000/    (Batch B - 50 videos)
    └── ...
```

Each cohort completely isolated - no data mixing.

---

## Combining Multiple Cohorts for ML

```python
from pathlib import Path
import pandas as pd

# Load all cohort CSVs
cohort_files = Path('data').glob('cohort_*/final_dataset.csv')
dfs = [pd.read_csv(f) for f in cohort_files]

# Combine
combined = pd.concat(dfs, ignore_index=True)

print(f"Videos: {combined['video_id'].nunique()}")
print(f"Rows: {len(combined)}")
```

Result: 100s of videos tracked over time for ML training.