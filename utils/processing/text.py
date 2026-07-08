import re


def tweet_to_tokens(tweet):
    """
    Text preprocessing (designed for tweets):
    1. removes: 
        - stock tickers like $GE
        - old-style retweet markers "RT"
        - hyperlinks
        - "#" symbols
    2. tokenization via TweetTokenizer:
        - with lowercasing
        - with removal of Twitter handles (example: @katyperry)
        - with replacement of repeated character sequences of length 3 or more with sequences 
            of length 3 (example: waaaaaayyyy -> waaayyy)
    """
    from nltk.tokenize import TweetTokenizer
    
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    return tokenizer.tokenize(tweet)
