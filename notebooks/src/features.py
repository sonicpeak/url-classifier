import pandas as pd, re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split

SUS = ["login", "verify", "bank", "secure", "account", "update", "signin", "paypal"]
IP_RE = re.compile(r"(\d{1,3}\.){3}\d{1,3}")

def extract(url):
    u = url if "://" in url else "http://" + url
    try:
        host = urlparse(u).netloc
    except ValueError:
        host = u.split("://", 1)[1].split("/", 1)[0]
    return {
        "url_len": len(url),
        "host_len": len(host),
        "dots": url.count("."),
        "hyphens": url.count("-"),
        "has_at": int("@" in url),
        "has_ip": int(bool(IP_RE.search(host))),
        "https": int(url.lower().startswith("https")),
        "subdomains": max(host.count(".") - 1, 0),
        "digits": sum(c.isdigit() for c in url),
        "slashes": url.count("/"),
        "sus_words": sum(w in url.lower() for w in SUS),
    }

def load_data(path="data/phishing_site_urls.csv"):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="URL").reset_index(drop=True)
    df["y"] = (df.Label == "bad").astype(int)
    X = pd.DataFrame(df.URL.apply(extract).tolist())
    return df, X, df.y

def split(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

if __name__ == "__main__":
    df, X, y = load_data()
    print(X.shape)
    print(X.head())