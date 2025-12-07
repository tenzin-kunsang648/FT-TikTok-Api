"""
=============================================================================
TikTok Time-Series Video Tracker
=============================================================================

PURPOSE:
    Track the same TikTok videos at multiple timepoints to build dataset
    for ML virality prediction model.

HOW IT WORKS:
    1. Collect trending videos, select newest available
    2. Save those video IDs
    3. Re-fetch THOSE SAME videos at later timepoints
    4. Calculate velocity metrics (growth rates)
    5. Export time-series CSV for ML training

NOTE ON VIDEO FRESHNESS:
    TikTok's trending API returns videos that may be hours or days old.
    The system selects the NEWEST available from trending, then tracks
    those videos over time. The timepoints (hour_1, hour_6, etc.) represent
    time elapsed since FIRST COLLECTION, not since upload.

WORKFLOW:
    Session 1:  python3 tiktok_tracker.py init 50     (collect 50 videos)
    +1 hour:    python3 tiktok_tracker.py collect     (re-fetch same 50)
    +6 hours:   python3 tiktok_tracker.py collect     (re-fetch same 50)
    +24 hours:  python3 tiktok_tracker.py collect     (re-fetch same 50)
    +48 hours:  python3 tiktok_tracker.py collect     (re-fetch same 50)
                python3 tiktok_tracker.py export      (create CSV)

COMMANDS:
    init [count]         - Start tracking N videos
    collect [cohort_id]  - Re-collect tracked videos
    export [cohort_id]   - Export time-series CSV
    status [cohort_id]   - Show progress
    list                 - Show all cohorts

OUTPUT:
    data/cohort_YYYYMMDD_HHMMSS/final_dataset.csv
    Format: N videos × 5 timepoints = N×5 rows

=============================================================================
"""

import asyncio
import json
import csv
from pathlib import Path
from datetime import datetime
from TikTokApi import TikTokApi
import sys


# =============================================================================
# UTILITIES
# =============================================================================

def safe_int(value, default=0):
    """Convert to int safely, handling None and strings"""
    if value is None:
        return default
    try:
        return int(value) if not isinstance(value, str) else int(value)
    except:
        return default


# =============================================================================
# COHORT MANAGER
# =============================================================================

class CohortManager:
    """Manages video cohorts and tracking state"""
    
    def __init__(self, cohort_id=None):
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        if cohort_id:
            self.cohort_id = cohort_id
        else:
            self.cohort_id = self._find_active()
        
        if self.cohort_id:
            self.cohort_dir = self.data_dir / f'cohort_{self.cohort_id}'
            self.tracking_file = self.cohort_dir / 'tracked_videos.json'
        else:
            self.cohort_dir = None
            self.tracking_file = None
    
    def _find_active(self):
        """Find most recent incomplete cohort"""
        cohorts = sorted(self.data_dir.glob('cohort_*'), reverse=True)
        
        for cohort_dir in cohorts:
            tracking_file = cohort_dir / 'tracked_videos.json'
            if tracking_file.exists():
                with open(tracking_file, 'r') as f:
                    data = json.load(f)
                if not data.get('collections', {}).get('hour_48'):
                    return cohort_dir.name.replace('cohort_', '')
        return None
    
    def create(self, video_ids):
        """Create new cohort"""
        self.cohort_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.cohort_dir = self.data_dir / f'cohort_{self.cohort_id}'
        self.cohort_dir.mkdir(exist_ok=True)
        self.tracking_file = self.cohort_dir / 'tracked_videos.json'
        
        with open(self.tracking_file, 'w') as f:
            json.dump({
                'cohort_id': self.cohort_id,
                'created_at': datetime.now().isoformat(),
                'video_ids': video_ids,
                'collections': {}
            }, f, indent=2)
        
        return self.cohort_id
    
    def get_video_ids(self):
        """Get tracked video IDs"""
        if not self.tracking_file or not self.tracking_file.exists():
            return []
        with open(self.tracking_file, 'r') as f:
            return json.load(f).get('video_ids', [])
    
    def save_snapshot(self, videos, timepoint):
        """Save snapshot and update tracking"""
        with open(self.cohort_dir / f'{timepoint}.json', 'w', encoding='utf-8') as f:
            json.dump(videos, f, indent=2, ensure_ascii=False, default=str)
        
        with open(self.tracking_file, 'r') as f:
            data = json.load(f)
        data['collections'][timepoint] = {
            'collected_at': datetime.now().isoformat(),
            'count': len(videos)
        }
        with open(self.tracking_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def next_timepoint(self):
        """Get next uncollected timepoint"""
        if not self.tracking_file.exists():
            return None
        with open(self.tracking_file, 'r') as f:
            collections = json.load(f).get('collections', {})
        for tp in ['hour_1', 'hour_6', 'hour_24', 'hour_48']:
            if tp not in collections:
                return tp
        return None


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_data(raw):
    """Extract all fields from raw video data"""
    author = raw.get('author', {})
    author_stats = raw.get('authorStats', {})
    stats = raw.get('statsV2', raw.get('stats', {}))
    music = raw.get('music', {})
    video = raw.get('video', {})
    
    upload_ts = raw.get('createTime')
    if upload_ts:
        try:
            upload_time = datetime.fromtimestamp(int(upload_ts))
            hours_since = (datetime.now() - upload_time).total_seconds() / 3600
            upload_dt = upload_time.isoformat()
        except:
            hours_since = None
            upload_dt = None
    else:
        hours_since = None
        upload_dt = None
    
    views = safe_int(stats.get('playCount'))
    likes = safe_int(stats.get('diggCount'))
    comments = safe_int(stats.get('commentCount'))
    shares = safe_int(stats.get('shareCount'))
    
    return {
        'video_id': str(raw.get('id')),
        'video_url': f"https://www.tiktok.com/@{author.get('uniqueId', 'unknown')}/video/{raw.get('id')}",
        'upload_timestamp': safe_int(upload_ts),
        'upload_datetime': upload_dt,
        'hours_since_upload': round(hours_since, 2) if hours_since else None,
        'collection_timestamp': datetime.now().isoformat(),
        'caption': str(raw.get('desc', '')),
        'hashtags': ', '.join([c.get('title', '') for c in raw.get('challenges', [])]),
        'video_duration': safe_int(video.get('duration')),
        'thumbnail_url': str(video.get('cover', '')),
        'music_id': str(music.get('id', '')),
        'sound_title': str(music.get('title', '')),
        'sound_author': str(music.get('authorName', '')),
        'sound_original': bool(music.get('original', False)),
        'views': views,
        'likes': likes,
        'comments_count': comments,
        'shares': shares,
        'saves': safe_int(stats.get('collectCount')),
        'engagement_rate': round((likes / views * 100), 4) if views > 0 else 0,
        'comment_rate': round((comments / views * 1000), 4) if views > 0 else 0,
        'share_rate': round((shares / views * 1000), 4) if views > 0 else 0,
        'creator_username': str(author.get('uniqueId', '')),
        'creator_display_name': str(author.get('nickname', '')),
        'creator_verified': bool(author.get('verified', False)),
        'creator_follower_count': safe_int(author_stats.get('followerCount')),
        'creator_following_count': safe_int(author_stats.get('followingCount')),
        'creator_total_likes': safe_int(author_stats.get('heartCount')),
        'creator_total_videos': safe_int(author_stats.get('videoCount')),
        'creator_bio': str(author.get('signature', '')),
    }


# =============================================================================
# INIT
# =============================================================================

async def init_cohort(count=100):
    """
    Initialize tracking: collect videos and save IDs.
    
    Collects from trending, sorts by upload time, selects newest N.
    """
    api = None
    
    try:
        print("\n" + "=" * 70)
        print("INIT: Starting New Cohort")
        print("=" * 70)
        print(f"\nTarget: {count} videos (selecting newest from trending)\n")
        
        api = TikTokApi()
        print("⏳ Creating session...")
        await api.create_sessions(num_sessions=1, headless=False, browser="webkit")
        print("✓ Ready\n")
        
        # Collect from trending
        print(f"📊 Collecting {count*2} trending videos...")
        videos = []
        
        collected = 0
        async for video in api.trending.videos(count=count*2):
            videos.append(video.as_dict)
            collected += 1
            
            if collected % 10 == 0:
                print(f"  Progress: {collected}")
            
            if collected >= count*2:
                break
            
            await asyncio.sleep(0.5)
        
        print(f"  ✓ Collected {len(videos)}\n")
        
        if not videos:
            print("✗ No videos collected")
            return
        
        # Sort by upload time
        print(f"📅 Sorting by upload time...")
        with_time = []
        for v in videos:
            ts = v.get('createTime')
            if ts:
                try:
                    hours_ago = (datetime.now() - datetime.fromtimestamp(int(ts))).total_seconds() / 3600
                    with_time.append({'raw': v, 'ts': int(ts), 'hours': hours_ago})
                except:
                    continue
        
        with_time.sort(key=lambda x: x['ts'], reverse=True)
        newest = with_time[:count]
        
        if newest:
            print(f"  Newest: {newest[0]['hours']:.1f}h ago")
            print(f"  Oldest: {newest[-1]['hours']:.1f}h ago\n")
        
        # Extract IDs and data
        video_ids = [str(v['raw'].get('id')) for v in newest]
        hour_0_data = [extract_data(v['raw']) for v in newest]
        
        # Create cohort
        manager = CohortManager()
        cohort_id = manager.create(video_ids)
        manager = CohortManager(cohort_id)
        manager.save_snapshot(hour_0_data, 'hour_0')
        
        print("=" * 70)
        print(f"✓ COHORT: {cohort_id}")
        print("=" * 70)
        print(f"\n📁 data/cohort_{cohort_id}/")
        print(f"📊 Tracking: {len(video_ids)} videos")
        print(f"\n⏰ Next: python3 tiktok_tracker.py collect (in 1+ hours)")
        print("=" * 70 + "\n")
        
    finally:
        if api:
            await api.close_sessions()
            await api.stop_playwright()


# =============================================================================
# COLLECT
# =============================================================================

async def collect_timepoint(cohort_id=None):
    """Re-fetch tracked videos at next timepoint"""
    api = None
    
    try:
        manager = CohortManager(cohort_id)
        
        if not manager.cohort_id:
            print("\n✗ No cohort. Run: python3 tiktok_tracker.py init\n")
            return
        
        next_tp = manager.next_timepoint()
        if not next_tp:
            print(f"\n✓ Complete! Run: python3 tiktok_tracker.py export\n")
            return
        
        print(f"\n" + "=" * 70)
        print(f"COLLECT: {next_tp.upper()} - Cohort {manager.cohort_id}")
        print("=" * 70 + "\n")
        
        video_ids = manager.get_video_ids()
        print(f"Re-fetching {len(video_ids)} videos...\n")
        
        api = TikTokApi()
        await api.create_sessions(num_sessions=1, headless=False, browser="webkit")
        
        data = []
        for i, vid in enumerate(video_ids, 1):
            try:
                # Fetch using video ID - need to get data from trending first
                # Then re-fetch with full info
                video = api.video(id=vid)
                
                # Use make_request to get video data by ID
                params = {'itemId': vid}
                raw = await api.make_request(
                    url='https://www.tiktok.com/api/item/detail/',
                    params=params
                )
                
                if raw and raw.get('itemInfo'):
                    video_data = extract_data(raw['itemInfo']['itemStruct'])
                    data.append(video_data)
                    print(f"  [{i}/{len(video_ids)}] {vid}: {video_data['views']:,} views")
                else:
                    print(f"  ✗ {vid}: No data returned")
                
                await asyncio.sleep(2)
            except Exception as e:
                print(f"  ✗ {vid}: {e}")
        
        manager.save_snapshot(data, next_tp)
        print(f"\n✓ Saved: {next_tp}.json")
        
        if manager.next_timepoint():
            print(f"⏰ Next: {manager.next_timepoint()}\n")
        else:
            print(f"✅ Done! Run: python3 tiktok_tracker.py export\n")
        
    finally:
        if api:
            await api.close_sessions()
            await api.stop_playwright()


# =============================================================================
# EXPORT
# =============================================================================

def export_dataset(cohort_id=None):
    """Export time-series CSV with velocities"""
    manager = CohortManager(cohort_id)
    
    if not manager.cohort_id:
        print("\n✗ No cohort\n")
        return
    
    print(f"\n" + "=" * 70)
    print(f"EXPORT: {manager.cohort_id}")
    print("=" * 70 + "\n")
    
    # Load all snapshots
    all_rows = []
    for tp in ['hour_0', 'hour_1', 'hour_6', 'hour_24', 'hour_48']:
        f = manager.cohort_dir / f'{tp}.json'
        if f.exists():
            with open(f, 'r') as file:
                videos = json.load(file)
                for v in videos:
                    v['timepoint'] = tp
                    v['cohort_id'] = manager.cohort_id
                all_rows.extend(videos)
                print(f"  ✓ {tp}: {len(videos)}")
        else:
            print(f"  ⚠️  {tp}: missing")
    
    if not all_rows:
        print("\n✗ No data\n")
        return
    
    print(f"\n✓ {len(all_rows)} rows")
    print("⏳ Calculating velocities...\n")
    
    # Group by video and calculate velocities
    by_vid = {}
    for row in all_rows:
        vid = row['video_id']
        if vid not in by_vid:
            by_vid[vid] = []
        by_vid[vid].append(row)
    
    enriched = []
    for vid, snaps in by_vid.items():
        snaps.sort(key=lambda x: x.get('hours_since_upload', 0))
        
        for i, snap in enumerate(snaps):
            if i > 0:
                prev = snaps[i-1]
                h_diff = snap.get('hours_since_upload', 0) - prev.get('hours_since_upload', 0)
                
                if h_diff > 0:
                    snap['views_velocity'] = round((snap['views'] - prev['views']) / h_diff, 2)
                    snap['likes_velocity'] = round((snap['likes'] - prev['likes']) / h_diff, 2)
                    snap['views_growth_pct'] = round(((snap['views'] - prev['views']) / prev['views'] * 100), 2) if prev['views'] > 0 else 0
                else:
                    snap['views_velocity'] = snap['likes_velocity'] = snap['views_growth_pct'] = 0
            else:
                snap['views_velocity'] = snap['likes_velocity'] = snap['views_growth_pct'] = 0
            
            enriched.append(snap)
    
    # Export CSV
    out = manager.cohort_dir / 'final_dataset.csv'
    
    headers = [
        # Identifiers
        'cohort_id', 'video_id', 'timepoint', 'video_url',
        
        # Timing
        'upload_timestamp', 'upload_datetime', 'collection_timestamp', 'hours_since_upload',
        
        # Content
        'caption', 'hashtags', 'video_duration', 'thumbnail_url',
        
        # Music/Sound
        'music_id', 'sound_title', 'sound_author', 'sound_original',
        
        # Engagement (raw)
        'views', 'likes', 'comments_count', 'shares', 'saves',
        
        # Engagement (calculated)
        'engagement_rate', 'comment_rate', 'share_rate',
        
        # Velocity (change over time)
        'views_velocity', 'likes_velocity', 'views_growth_pct',
        
        # Creator
        'creator_username', 'creator_display_name', 'creator_verified',
        'creator_follower_count', 'creator_following_count',
        'creator_total_likes', 'creator_total_videos', 'creator_bio'
    ]
    
    with open(out, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, headers, extrasaction='ignore').writeheader()
        csv.DictWriter(f, headers, extrasaction='ignore').writerows(enriched)
    
    complete = sum(1 for s in by_vid.values() if len(s) == 5)
    
    print(f"✓ CSV: {out}")
    print(f"\n📊 {len(by_vid)} videos | {complete} complete | {len(enriched)} rows\n")


# =============================================================================
# STATUS
# =============================================================================

def list_cohorts():
    """List all cohorts"""
    cohorts = sorted(Path('data').glob('cohort_*')) if Path('data').exists() else []
    
    if not cohorts:
        print("\n✗ No cohorts\n")
        return
    
    print(f"\n" + "=" * 70)
    print(f"COHORTS ({len(cohorts)})")
    print("=" * 70 + "\n")
    
    for c in cohorts:
        cid = c.name.replace('cohort_', '')
        t = c / 'tracked_videos.json'
        
        if t.exists():
            with open(t, 'r') as f:
                d = json.load(f)
            
            vids = len(d.get('video_ids', []))
            done = len([x for x in ['hour_0', 'hour_1', 'hour_6', 'hour_24', 'hour_48'] if x in d.get('collections', {})])
            status = "✅" if done == 5 else f"⏳ {done}/5"
            
            print(f"📁 {cid} | {vids} videos | {status}")
        
        print()
    
    print("=" * 70 + "\n")


def show_status(cohort_id=None):
    """Show cohort status"""
    manager = CohortManager(cohort_id)
    
    if not manager.cohort_id:
        print("\n✗ No cohort\n")
        return
    
    with open(manager.tracking_file, 'r') as f:
        d = json.load(f)
    
    print(f"\n" + "=" * 70)
    print(f"COHORT: {manager.cohort_id}")
    print("=" * 70)
    print(f"\n📁 data/cohort_{manager.cohort_id}/")
    print(f"📊 {len(d.get('video_ids', []))} videos")
    print(f"\n📋 Collections:")
    
    for tp in ['hour_0', 'hour_1', 'hour_6', 'hour_24', 'hour_48']:
        if tp in d.get('collections', {}):
            print(f"   ✓ {tp}")
        else:
            print(f"   ⏳ {tp}")
    
    print(f"\n📅 Next: {manager.next_timepoint() or 'export'}")
    print("=" * 70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    if len(sys.argv) < 2:
        print("\nCommands:")
        print("  init [count]         Start tracking")
        print("  collect [cohort_id]  Collect next")
        print("  export [cohort_id]   Export CSV")
        print("  status [cohort_id]   Show status")
        print("  list                 List all\n")
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'init':
        await init_cohort(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif cmd == 'collect':
        await collect_timepoint(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == 'export':
        export_dataset(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == 'status':
        show_status(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == 'list':
        list_cohorts()
    else:
        print(f"\n✗ Unknown: {cmd}\n")


if __name__ == '__main__':
    asyncio.run(main())