import requests


""""
Use alpha vantage api

What do these mean
sentiment_score_definition: x <= -0.35: Bearish; 
-0.35 < x <= -0.15: Somewhat-Bearish; 
-0.15 < x < 0.15: Neutral; 
0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish
relevance_score_definition': '0 < x <= 1, with a higher score indicating higher relevance
"""


url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&topics=technology&apikey=YOUR_KEY'
r = requests.get(url)
data = r.json()

print(data['feed'][0])
#print(data['feed'])

