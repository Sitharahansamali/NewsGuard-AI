import os

from dotenv import load_dotenv
from newsapi import NewsApiClient

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

newsapi = NewsApiClient(api_key=API_KEY)


def get_latest_news():

    articles = newsapi.get_top_headlines(
        language="en",
        page_size=10
    )

    return articles["articles"]