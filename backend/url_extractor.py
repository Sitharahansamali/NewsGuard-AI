from newspaper import Article

def extract_news_from_url(url):

    article = Article(url)

    article.download()

    article.parse()

    return {
        "title": article.title,
        "text": article.text
    }