# TikTok Time-Series Video Tracker

Track the same TikTok videos over time for ML virality prediction.

Built using [TikTok-Api](https://github.com/davidteather/TikTok-Api) by David Teather.

---

## What It Does

1. Collects trending videos, selects newest available
2. Saves those video IDs
3. Re-fetches THE SAME videos at intervals
4. Calculates velocity metrics (growth rates)
5. Exports time-series CSV for ML training

**Important:** Timepoints (hour_1, hour_6, etc.) represent time elapsed since **first collection**, not since video upload. TikTok's trending API returns videos that may already be hours or days old.

---

## Quick Start

```bash
# Install dependencies
pip install TikTokApi playwright
playwright install webkit

# Start tracking 50 videos
python3 FT/tiktok_tracker.py init 50

# Collect at intervals (set reminders for 1h, 6h, 24h, 48h later)
python3 FT/tiktok_tracker.py collect

# Export final dataset
python3 FT/tiktok_tracker.py export
```

---

## Commands

```bash
python3 tiktok_tracker.py init [count]         # Start tracking N videos (default: 100)
python3 tiktok_tracker.py collect [cohort_id]  # Re-collect tracked videos
python3 tiktok_tracker.py export [cohort_id]   # Export time-series CSV
python3 tiktok_tracker.py status [cohort_id]   # Show progress
python3 tiktok_tracker.py list                 # List all cohorts
```

---

## Manual Workflow

Set 4 reminders over 48 hours:

**Monday 12pm - Initialize:**
```bash
python3 tiktok_tracker.py init 50
```
Collects 50 videos, saves as hour_0.

**Monday 1pm (+1 hour):**
```bash
python3 tiktok_tracker.py collect
```
Re-fetches SAME 50 videos, saves as hour_1.

**Monday 6pm (+6 hours):**
```bash
python3 tiktok_tracker.py collect
```
Re-fetches SAME 50 videos, saves as hour_6.

**Tuesday 12pm (+24 hours):**
```bash
python3 tiktok_tracker.py collect
```
Re-fetches SAME 50 videos, saves as hour_24.

**Wednesday 12pm (+48 hours):**
```bash
python3 tiktok_tracker.py collect
python3 tiktok_tracker.py export
```
Re-fetches SAME 50 videos (hour_48), exports final CSV.

---

## File Organization

```
project/
├── README.md                          (this file)
│
├── FT/
│   ├── tiktok_tracker.py              (main script)
│   │
│   ├── data/                          (auto-created)
│   │   └── cohort_YYYYMMDD_HHMMSS/
│   │       ├── tracked_videos.json
│   │       ├── hour_0.json ... hour_48.json
│   │       └── final_dataset.csv
│   │
│   └── documentation/
│       ├── MULTI_COHORT_GUIDE.md
│       └── Implementation_Comparison.md
```

---

## Managing Multiple Cohorts

Can run multiple batches in parallel:

```bash
# List all cohorts
python3 tiktok_tracker.py list

# Collect specific cohort
python3 tiktok_tracker.py collect 20251207_120000
python3 tiktok_tracker.py collect 20251214_120000

# Export specific cohort
python3 tiktok_tracker.py export 20251207_120000
```

See [MULTI_COHORT_GUIDE.md](FT/documentation/MULTI_COHORT_GUIDE.md) for detailed examples.

---

## Output CSV Format

Each row = one video at one timepoint

```csv
cohort_id,video_id,timepoint,views,likes,engagement_rate,views_velocity,...
20251207_120000,123,hour_0,100000,5000,5.0,0,...
20251207_120000,123,hour_1,105000,5250,5.0,5000,...
20251207_120000,123,hour_6,120000,6000,5.0,3000,...
20251207_120000,123,hour_24,180000,9000,5.0,3333,...
20251207_120000,123,hour_48,220000,11000,5.0,1666,...
```

**50 videos × 5 timepoints = 250 rows**

**Columns (40):**
- Identifiers: cohort_id, video_id, timepoint, video_url
- Timing: upload_datetime, collection_timestamp, hours_since_upload
- Engagement (raw): views, likes, comments_count, shares, saves
- Engagement (calculated): engagement_rate, comment_rate, share_rate
- Velocity: views_velocity, likes_velocity, views_growth_pct
- Content: caption, hashtags, video_duration, thumbnail_url
- Music: music_id, sound_title, sound_author, sound_original
- Creator: username, display_name, follower_count, verified, bio
- Comment stats: top_comment_text, top_comment_likes, avg_comment_likes

---

## For ML Model Training

```python
import pandas as pd

# Load dataset
df = pd.read_csv('FT/data/cohort_20251207_120000/final_dataset.csv')

# Features from hour_1 (early signals)
X_train = df[df['timepoint'] == 'hour_1'][[
    'views', 'likes', 'engagement_rate', 'views_velocity',
    'creator_follower_count', 'creator_verified', 'video_duration'
]]

# Target from hour_24 (predict future performance)
y_train = df[df['timepoint'] == 'hour_24']['views']

# Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

---

## Timeline

**Total time needed:** 48 hours  
**Active time:** ~2 hours (5 sessions × ~20-30 min)  
**Manual interventions:** 5 commands

Laptop can be turned off between collections.

---

## Data Coverage

**Available (40 fields collected):**
- All video metadata
- All engagement metrics  
- All creator information
- Calculated rates and velocities
- Comment statistics (top comment, averages)

**Not Available:**
- duet_count, stitch_count (not in TikTok public API)
- completion_rate, watch_time (requires TikTok Analytics - creator access only)
- traffic_source, FYP appearances (internal TikTok metrics)

See [Implementation_Comparison.md](FT/documentation/Implementation_Comparison.md) for complete field analysis.

---

## Attribution

This project is built using the [TikTok-Api](https://github.com/davidteather/TikTok-Api) library by David Teather.

---

## Documentation

- [Managing multiple batches simultaneously](FT/documentation/MULTI_COHORT_GUIDE.md) 
- [Complete comparison of requirements vs. implementation](FT/documentation/Implementation_Comparison.md) 

---

## License

MIT License