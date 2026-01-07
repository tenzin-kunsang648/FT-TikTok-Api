"""
TikTok Tracker using Apify
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from apify_client import ApifyClient
from data_collect_config import MY_SECRET_APIFY_TOKEN


# =============================================================================
# CONFIGURATION
# =============================================================================

APIFY_TOKEN = os.getenv('APIFY_TOKEN', MY_SECRET_APIFY_TOKEN)

VIDEOS_PER_COHORT = 500


# =============================================================================
# REGISTRY & COHORT MANAGER (same as before)
# =============================================================================

class VideoRegistry:
    """Track all videos across all cohorts to prevent duplicates"""
    
    def __init__(self):
        self.registry_file = Path('data') / 'global_video_registry.json'
        self.registry_file.parent.mkdir(exist_ok=True)
        self._load()
    
    def _load(self):
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {'tracked_videos': {}}
    
    def _save(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_videos(self, video_ids, cohort_id):
        for vid in video_ids:
            self.data['tracked_videos'][str(vid)] = {
                'cohort_id': cohort_id,
                'added_at': datetime.now().isoformat()
            }
        self._save()
    
    def is_tracked(self, video_id):
        return str(video_id) in self.data['tracked_videos']
    
    def filter_new(self, video_ids):
        return [v for v in video_ids if not self.is_tracked(v)]


class CohortManager:
    """Manages video cohorts and tracking state"""
    
    def __init__(self, cohort_id=None):
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        if cohort_id:
            self.cohort_id = cohort_id
            self.cohort_dir = self.data_dir / f'cohort_{self.cohort_id}'
            self.tracking_file = self.cohort_dir / 'tracked_videos.json'
        else:
            self.cohort_dir = None
            self.tracking_file = None
    
    def create(self, video_ids):
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
    
    def save_snapshot(self, videos, timepoint):
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


# =============================================================================
# APIFY SCRAPER
# =============================================================================

def scrape_trending_videos(count=100):
    """
    Scrape trending videos using Apify
    Returns list of video data dictionaries
    """
    print(f"\n🚀 Starting Apify scraper...")
    print(f"   Target: {count} videos\n")
    
    if APIFY_TOKEN == 'YOUR_TOKEN_HERE':
        print("❌ ERROR: Please set your Apify token!")
        print("\n1. Go to: https://console.apify.com/account/integrations")
        print("2. Copy your API token")
        print("3. Set it:")
        print("   export APIFY_TOKEN='your_token_here'")
        print("   OR")
        print("   Edit this file and replace 'YOUR_TOKEN_HERE'\n")
        return []
    
    client = ApifyClient(APIFY_TOKEN)
    
    # Try different scraper - this one is more reliable
    # Using hashtag search for #fyp to get trending content
    run_input = {
        "hashtags": ["fyp", "foryou", "trending"],  # Get trending content
        "resultsPerPage": count,
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
    }
    
    try:
        print("⏳ Running Apify actor (this takes 1-2 minutes)...")
        print("   Using hashtag search: #fyp #foryou #trending\n")
        
        # Try the main TikTok scraper instead
        run = client.actor("clockworks/tiktok-scraper").call(run_input=run_input)
        
        print("✓ Scraper finished!")
        print("⏳ Fetching results...\n")
        
        videos = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            videos.append(item)
        
        print(f"✓ Retrieved {len(videos)} videos\n")
        
        if len(videos) == 0:
            print("⚠️  No videos returned. Trying alternative method...\n")
            # Try without hashtags - just trending
            run_input2 = {
                "resultsPerPage": count,
                "shouldDownloadVideos": False,
            }
            run2 = client.actor("clockworks/tiktok-scraper").call(run_input=run_input2)
            
            videos = []
            for item in client.dataset(run2["defaultDatasetId"]).iterate_items():
                videos.append(item)
            
            print(f"✓ Retrieved {len(videos)} videos (alternative method)\n")
        
        return videos
        
    except Exception as e:
        print(f"\n❌ Apify error: {e}\n")
        
        # Try the backup free scraper with different config
        print("⚠️  Trying backup scraper...\n")
        try:
            run_input3 = {
                "profiles": ["https://www.tiktok.com/@tiktok"],  # Official TikTok account
                "resultsLimit": count,
            }
            run3 = client.actor("clockworks/free-tiktok-scraper").call(run_input=run_input3)
            
            videos = []
            for item in client.dataset(run3["defaultDatasetId"]).iterate_items():
                videos.append(item)
            
            print(f"✓ Retrieved {len(videos)} videos (backup method)\n")
            return videos
        except Exception as e2:
            print(f"❌ Backup also failed: {e2}\n")
            print("Possible issues:")
            print("  - Out of credits (check: https://console.apify.com/billing)")
            print("  - Scraper configuration issue")
            print("  - Try manual method instead\n")
            return []


def extract_data(raw):
    """Extract relevant fields from Apify video data"""
    
    # Calculate hours since upload
    create_time = raw.get('createTime', raw.get('createTimeISO'))
    if create_time:
        try:
            if isinstance(create_time, str):
                upload_time = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
            else:
                upload_time = datetime.fromtimestamp(int(create_time))
            hours_since = (datetime.now() - upload_time.replace(tzinfo=None)).total_seconds() / 3600
            upload_dt = upload_time.isoformat()
        except:
            hours_since = None
            upload_dt = None
    else:
        hours_since = None
        upload_dt = None
    
    author = raw.get('authorMeta', raw.get('author', {}))
    stats = raw.get('stats', {})
    music = raw.get('musicMeta', raw.get('music', {}))
    
    views = int(stats.get('playCount', 0))
    likes = int(stats.get('diggCount', 0))
    comments = int(stats.get('commentCount', 0))
    shares = int(stats.get('shareCount', 0))
    
    return {
        'video_id': str(raw.get('id', raw.get('video_id', ''))),
        'video_url': raw.get('webVideoUrl', raw.get('video_url', '')),
        'upload_timestamp': int(create_time) if isinstance(create_time, (int, float)) else None,
        'upload_datetime': upload_dt,
        'caption': str(raw.get('text', raw.get('desc', ''))),
        'hashtags': ', '.join([tag.get('name', '') for tag in raw.get('hashtags', [])]),
        'video_duration': int(raw.get('videoMeta', {}).get('duration', 0)),
        'collection_timestamp': datetime.now().isoformat(),
        'hours_since_upload': round(hours_since, 2) if hours_since else None,
        'views': views,
        'likes': likes,
        'comments_count': comments,
        'shares': shares,
        'saves': int(stats.get('collectCount', 0)),
        'engagement_rate': round((likes / views * 100), 4) if views > 0 else 0,
        'comment_rate': round((comments / views * 1000), 4) if views > 0 else 0,
        'share_rate': round((shares / views * 1000), 4) if views > 0 else 0,
        'creator_username': str(author.get('name', author.get('uniqueId', ''))),
        'creator_display_name': str(author.get('nickName', author.get('nickname', ''))),
        'creator_verified': bool(author.get('verified', False)),
        'creator_follower_count': int(author.get('fans', author.get('followerCount', 0))),
        'music_id': str(music.get('musicId', '')),
        'sound_title': str(music.get('musicName', '')),
        'sound_author': str(music.get('musicAuthor', '')),
    }


# =============================================================================
# INIT COHORT WITH APIFY
# =============================================================================

def init_cohort(count=100):
    """Initialize cohort using Apify scraper"""
    
    print("\n" + "=" * 70)
    print("INIT: Starting New Cohort (via Apify)")
    print("=" * 70)
    print(f"\nTarget: {count} videos\n")
    
    registry = VideoRegistry()
    
    # Scrape videos from Apify
    raw_videos = scrape_trending_videos(count)
    
    if not raw_videos:
        print("✗ No videos scraped\n")
        return
    
    # Extract data
    print("📊 Processing video data...")
    all_videos = []
    for raw in raw_videos:
        try:
            data = extract_data(raw)
            if data['video_id']:
                all_videos.append(data)
        except Exception as e:
            print(f"  ⚠️  Error processing video: {e}")
    
    print(f"  Processed: {len(all_videos)} videos\n")
    
    if not all_videos:
        print("✗ No valid videos\n")
        return
    
    # Filter duplicates
    print(f"🔍 Filtering duplicates...")
    video_ids_all = [v['video_id'] for v in all_videos]
    new_video_ids = registry.filter_new(video_ids_all)
    
    print(f"  Total collected: {len(video_ids_all)}")
    print(f"  Already tracked: {len(video_ids_all) - len(new_video_ids)}")
    print(f"  Available new: {len(new_video_ids)}\n")
    
    # Filter to only new videos
    videos = [v for v in all_videos if v['video_id'] in new_video_ids]
    
    if not videos:
        print("✗ No new videos to track\n")
        return
    
    # Sort by upload time (newest first)
    print(f"📅 Sorting by upload time...")
    videos_with_time = [v for v in videos if v.get('hours_since_upload') is not None]
    videos_with_time.sort(key=lambda x: x['hours_since_upload'])
    
    # Take the newest N
    final_videos = videos_with_time[:count]
    
    if final_videos:
        print(f"  Newest: {final_videos[0]['hours_since_upload']:.1f}h ago")
        print(f"  Oldest: {final_videos[-1]['hours_since_upload']:.1f}h ago\n")
    
    # Create cohort
    video_ids = [v['video_id'] for v in final_videos]
    
    manager = CohortManager()
    cohort_id = manager.create(video_ids)
    manager = CohortManager(cohort_id)
    manager.save_snapshot(final_videos, 'hour_0')
    
    # Register videos
    registry.add_videos(video_ids, cohort_id)
    
    print("=" * 70)
    print(f"✓ COHORT: {cohort_id}")
    print("=" * 70)
    print(f"\n📁 data/cohort_{cohort_id}/")
    print(f"📊 Tracking: {len(video_ids)} videos")
    print(f"\n⏰ Next: python3 tiktok_tracker.py collect (in 1+ hours)")
    print("=" * 70 + "\n")


# =============================================================================
# CHECK CREDITS
# =============================================================================

def check_credits():
    """Check remaining Apify credits"""
    print("\n" + "=" * 70)
    print("APIFY CREDITS CHECK")
    print("=" * 70 + "\n")
    
    if APIFY_TOKEN == 'YOUR_TOKEN_HERE':
        print("❌ Please set your Apify token first!\n")
        return
    
    try:
        client = ApifyClient(APIFY_TOKEN)
        user = client.user().get()
        
        print(f"👤 User: {user.get('username', 'N/A')}")
        
        # Check different credit fields
        usage_credits = user.get('usageCredits', 0)
        monthly_credits = user.get('monthlyCredits', 0)
        free_credits = user.get('freeUsageCredits', 0)
        
        total = usage_credits + monthly_credits + free_credits
        
        print(f"💰 Available Credits:")
        print(f"   Usage credits: ${usage_credits:.2f}")
        print(f"   Monthly credits: ${monthly_credits:.2f}")
        print(f"   Free credits: ${free_credits:.2f}")
        print(f"   TOTAL: ${total:.2f}")
        
        print(f"\n📊 Estimate:")
        if total > 0:
            videos = int(total * 100)
            cohorts = int(videos / VIDEOS_PER_COHORT)
            print(f"   ~{videos} videos can be scraped")
            print(f"   = {cohorts} cohorts of {VIDEOS_PER_COHORT} videos")
        else:
            print(f"   ⚠️  No credits available")
            print(f"   Add credits at: https://console.apify.com/billing")
        
        print(f"\n🔗 View full details: https://console.apify.com/billing")
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        print("Full error details:")
        import traceback
        traceback.print_exc()
        print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("TIKTOK TRACKER - APIFY VERSION")
        print("=" * 70)
        print("\nCommands:")
        print("  python3 apify_tracker.py init [count]    - Start tracking (default: 100)")
        print("  python3 apify_tracker.py credits         - Check remaining credits")
        print("\nSetup:")
        print("  1. Get token: https://console.apify.com/account/integrations")
        print("  2. Set token: export APIFY_TOKEN='your_token_here'")
        print("\nAfter init, use regular tracker for collections:")
        print("  python3 tiktok_tracker.py collect")
        print("  python3 tiktok_tracker.py export")
        print("\n" + "=" * 70 + "\n")
        sys.exit(0)
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'init':
        count = int(sys.argv[2]) if len(sys.argv) > 2 else VIDEOS_PER_COHORT
        init_cohort(count)
    elif cmd == 'credits':
        check_credits()
    else:
        print(f"\n✗ Unknown command: {cmd}\n")