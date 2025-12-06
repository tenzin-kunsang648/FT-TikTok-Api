import json
import csv
from datetime import datetime

def json_to_csv(json_file, csv_file=None):
    """
    Convert TikTok JSON data to CSV format.
    
    Args:
        json_file: Path to the JSON file
        csv_file: Path to output CSV file (optional, auto-generated if not provided)
    """
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("✗ No data found in JSON file")
        return
    
    # Generate CSV filename if not provided
    if csv_file is None:
        csv_file = json_file.replace('.json', '.csv')
    
    # Define CSV columns (excluding raw_video_data to keep CSV readable)
    columns = [
        'video_id',
        'video_url',
        'upload_timestamp',
        'caption',
        'hashtags',
        'video_duration',
        'thumbnail_url',
        'music_id',
        'sound_title',
        'collection_timestamp',
        'views',
        'likes',
        'comments_count',
        'shares',
        'saves',
        'duet_count',
        'stitch_count',
        'creator_username',
        'creator_display_name',
        'creator_follower_count',
        'creator_following_count',
        'creator_total_likes',
        'creator_total_videos',
        'creator_verified',
        'creator_bio',
    ]
    
    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        
        for video in data:
            # Convert list to string for hashtags
            if isinstance(video.get('hashtags'), list):
                video['hashtags'] = ', '.join(video['hashtags'])
            
            # Convert timestamps to readable format
            if video.get('upload_timestamp'):
                try:
                    ts = int(video['upload_timestamp'])
                    video['upload_timestamp_readable'] = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass
            
            writer.writerow(video)
    
    print(f"✓ CSV file created: {csv_file}")
    print(f"✓ Converted {len(data)} videos")
    return csv_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Auto-detect the most recent JSON file in current directory
        import glob
        json_files = glob.glob("tiktok_videos_*.json")
        if json_files:
            json_file = max(json_files)  # Get most recent
            print(f"Using: {json_file}")
        else:
            print("✗ No JSON file found. Usage: python json_to_csv.py <json_file>")
            sys.exit(1)
    
    json_to_csv(json_file)