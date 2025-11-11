import requests
from requests.auth import HTTPBasicAuth
import praw
from datetime import datetime
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)


client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')

username = os.getenv('REDDIT_USERNAME')
password = os.getenv('REDDIT_PASSWORD')

data = {
    'grant_type': 'password',
    'username': username,
    'password': password
}

# POST 요청 헤더
headers = {
    'User-Agent': 'ydg06081'
}

# HTTP Basic 인증 (client_id와 client_secret 사용)
auth = HTTPBasicAuth(client_id, client_secret)

# POST 요청 보내기
response_auth = requests.post(
    'https://www.reddit.com/api/v1/access_token',
    headers=headers,
    data=data,
    auth=auth
)
#토큰 정보
token =  response_auth.json()['access_token']
# 요청 헤더
headers = {
    'Authorization': f'Bearer {token}'
}
import praw
from datetime import datetime

# Reddit 인증
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent="ydg06081"
)

# 특정 커뮤니티 지정
subreddit_name = "machinelearning"
subreddit = reddit.subreddit(subreddit_name)

# 최근 1개 글 수집
posts = []
for post in subreddit.new(limit=1):
    post_time = datetime.fromtimestamp(post.created_utc)
    posts.append({
        "title": post.title,
        "url": post.url,
        "created": post_time,
        "score": post.score,
        "num_comments": post.num_comments,
        "author": str(post.author),
        "selftext": post.selftext
    })

print(f"✅ {len(posts)}개의 글을 수집했습니다.")
print(posts[0]["selftext"])


