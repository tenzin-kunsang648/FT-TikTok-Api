# TikTok Time-Series Tracker

Track the same videos over time for ML virality prediction.

---

## What It Does

1. Collects trending videos, selects newest available
2. Saves those video IDs
3. Re-fetches THE SAME videos at intervals
4. Calculates velocity metrics (growth rates)
5. Exports time-series CSV for ML

**Note:** Timepoints (hour_1, hour_6, etc.) represent time elapsed since **first collection**, not since video upload. TikTok trending videos may already be hours/days old.

---

## Commands

```bash
python3 tiktok_tracker.py init [count]         # Start tracking N videos
python3 tiktok_tracker.py collect [cohort_id]  # Collect next timepoint
python3 tiktok_tracker.py export [cohort_id]   # Export CSV
python3 tiktok_tracker.py status [cohort_id]   # Show status  
python3 tiktok_tracker.py list                 # List cohorts
```

---

## Workflow Example

### Session 1 (Monday 12pm)
```bash
python3 tiktok_tracker.py init 50
```
Collects 50 videos, saves as hour_0

### Session 2 (Monday 1pm - 1 hour later)
```bash
python3 tiktok_tracker.py collect
```
Re-fetches SAME 50 videos, saves as hour_1

### Session 3 (Monday 6pm - 6 hours after Session 1)
```bash
python3 tiktok_tracker.py collect
```
Re-fetches SAME 50 videos, saves as hour_6

### Session 4 (Tuesday 12pm - 24 hours after Session 1)
```bash
python3 tiktok_tracker.py collect
```
Re-fetches SAME 50 videos, saves as hour_24

### Session 5 (Wednesday 12pm - 48 hours after Session 1)
```bash
python3 tiktok_tracker.py collect
python3 tiktok_tracker.py export
```
Re-fetches SAME 50 videos (hour_48), exports final CSV

---

## Multiple Cohorts

Start new cohorts anytime. Each is isolated.

```bash
# Week 1
python3 tiktok_tracker.py init 50  # Cohort A

# Week 2
python3 tiktok_tracker.py init 50  # Cohort B

# Manage both
python3 tiktok_tracker.py list
python3 tiktok_tracker.py collect 20251207_120000  # Cohort A
python3 tiktok_tracker.py collect 20251214_120000  # Cohort B
```

See `MULTI_COHORT_GUIDE.md` for details.

---

## Output CSV

```csv
cohort_id,video_id,timepoint,views,likes,engagement_rate,views_velocity
20251207_120000,123,hour_0,100000,5000,5.0,0
20251207_120000,123,hour_1,105000,5250,5.0,5000
20251207_120000,123,hour_6,120000,6000,5.0,3000
```

50 videos × 5 timepoints = 250 rows

---

## For ML

```python
df = pd.read_csv('data/cohort_20251207_120000/final_dataset.csv')

# Train on hour_1 → predict hour_24
X = df[df['timepoint'] == 'hour_1'][['views', 'likes', 'engagement_rate', 'views_velocity']]
y = df[df['timepoint'] == 'hour_24']['views']
```

---

## Installation

```bash
pip install TikTokApi playwright
playwright install webkit
```