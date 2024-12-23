import feedparser
import csv

# Define the RSS feeds
RSS_FEEDS = {
    # 'Copperbelt Energy': 'https://cecinvestor.com/search/kitwe/feed/rss2/',
    # 'ZNBC' : 'https://znbc.co.zm/news/search/kitwe/feed/rss2/',
    # 'News Invasion 24': 'https://newsinvasion24.com/search/kitwe/feed/rss2/',
    # 'Mwebantu': 'https://www.mwebantu.com/search/kitwe/feed/rss2/',
    'Lusaka Times': 'https://www.lusakatimes.com/search/kitwe/feed/rss2/',
    # 'Kitwe Online': 'https://kitweonline.com/search/kitwe/feed/rss2/',
    # 'Daily Revelation Zambia': 'https://dailyrevelationzambia.com/search/kitwe/feed/rss2/',
    # 'Zambia Monitor': 'https://www.zambiamonitor.com/search/kitwe/feed/rss2/',
    # 'Tech Africa News': 'https://www.techafricanews.com/search/kitwe/feed/rss2/',
    # 'Zambian Eye': 'https://zambianeye.com/search/kitwe/feed/rss2/',
    # 'DailyMail': 'https://www.daily-mail.co.zm/search/kitwe/feed/rss2/'
}

# Function to handle paginated feed entries
def get_feed_entries(feed_url, pages=10):
    all_entries = []
    for page in range(1, pages + 1):
        paged_url = f"{feed_url}?paged={page}"
        parsed_feed = feedparser.parse(paged_url)
        entries = parsed_feed.entries
        print(f'Collecting in page {page} for {feed_url}')
        if not entries:
            break  # Exit if there are no more entries
        all_entries.extend(entries)
    return all_entries

# CSV file path
csv_file_path = 'C:/Users/hoybr/sessionworkspace/Kitwe/News Collector/rss_feed_data2.csv'

print('Loading...')

# Open the CSV file for writing
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['Source', 'Category', 'Headline', 'Link', 'Description', 'Date'])
    
    # Loop through each RSS feed
    for source_name, feed_url in RSS_FEEDS.items():
        # Get paginated entries for the feed
        entries = get_feed_entries(feed_url, pages=50)
        print(f'RSS Feed done: {source_name}')
        
        # Process each entry and write to CSV
        for entry in entries:
            link = entry.link
            date = entry.published if 'published' in entry else 'N/A'
            description = entry.summary if 'summary' in entry else 'N/A'
            headline = entry.title if 'title' in entry else 'N/A'

            # Collect all categories/tags
            category = ', '.join(tag.term for tag in entry.tags) if 'tags' in entry and entry.tags else 'N/A'
            
            # Write the entry data to the CSV file
            writer.writerow([source_name, category, headline, link, description, date])

print(f"Data has been saved to {csv_file_path}")
