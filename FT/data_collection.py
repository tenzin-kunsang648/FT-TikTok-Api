import asyncio
from TikTokApi import TikTokApi
from datetime import datetime
import json
import traceback


def safe_get(data, *keys, default=None):
    """Safely navigate nested dictionaries"""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, {})
        else:
            return default
    return data if data != {} else default


async def main():
    api = None
    videos_data = []
    
    try:
        api = TikTokApi()
        print("✓ API initialized")
        
        await api.create_sessions(
            num_sessions=1,
            headless=False,
            browser="webkit"
        )
        print("✓ Sessions created")
        
        async for video in api.trending.videos(count=5):
            print(f"✓ Got video: {video.id}")
            
            # All data comes from video.as_dict
            v = video.as_dict
            author_data = safe_get(v, "author", default={})
            stats_data = safe_get(v, "statsV2", default=safe_get(v, "stats", default={}))
            music_data = safe_get(v, "music", default={})
            video_meta = safe_get(v, "video", default={})
            
            # Extract video metadata
            video_info = {
                # A. VIDEO METADATA
                "video_id": v.get("id"),
                "video_url": f"https://www.tiktok.com/@{author_data.get('uniqueId')}/video/{v.get('id')}",
                "upload_timestamp": v.get("createTime"),
                "caption": v.get("desc", ""),
                "hashtags": [tag.get("title") for tag in v.get("challenges", [])],
                "video_duration": video_meta.get("duration", 0),
                "thumbnail_url": v.get("video", {}).get("dynamicCover", ""),
                "music_id": music_data.get("id"),
                "sound_title": music_data.get("title"),
                
                # B. ENGAGEMENT METRICS
                "collection_timestamp": datetime.now().isoformat(),
                "views": stats_data.get("playCount", 0),
                "likes": stats_data.get("diggCount", 0),
                "comments_count": stats_data.get("commentCount", 0),
                "shares": stats_data.get("shareCount", 0),
                "saves": stats_data.get("collectCount", 0),
                "duet_count": safe_get(v, "duetInfo", "duetCount", default=0),
                "stitch_count": safe_get(v, "stitchInfo", "stitchCount", default=0),
                
                # C. CREATOR INFO
                "creator_username": author_data.get("uniqueId"),
                "creator_display_name": author_data.get("nickname"),
                "creator_follower_count": safe_get(author_data, "stats", "followerCount", default=0),
                "creator_following_count": safe_get(author_data, "stats", "followingCount", default=0),
                "creator_total_likes": safe_get(author_data, "stats", "heartCount", default=0),
                "creator_total_videos": safe_get(author_data, "stats", "videoCount", default=0),
                "creator_verified": author_data.get("verified", False),
                "creator_bio": author_data.get("signature", ""),
                
                # RAW DATA (for debugging)
                "raw_video_data": v,
            }
            
            videos_data.append(video_info)
            print(f"  ✓ Collected: {video_info['caption'][:50]}...")

        
        # Save to JSON file
        output_file = f"tiktok_videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(videos_data, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n✓ Data saved to {output_file}")
        print(f"✓ Collected {len(videos_data)} videos")
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if api:
            await api.close_sessions()
            await api.stop_playwright()

asyncio.run(main())