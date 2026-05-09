import tldextract


TRUSTED_SOURCES = {
    "bbc.com": 95,
    "reuters.com": 98,
    "cnn.com": 85,
    "nytimes.com": 90,
    "aljazeera.com": 88
}


def get_domain(url):

    extracted = tldextract.extract(url)

    return extracted.domain + "." + extracted.suffix


def check_source_credibility(domain):

    score = TRUSTED_SOURCES.get(domain, 50)

    if score >= 90:
        level = "High"

    elif score >= 70:
        level = "Medium"

    else:
        level = "Low"

    return {
        "score": score,
        "level": level
    }