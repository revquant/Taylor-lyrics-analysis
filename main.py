import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk, json, re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Load lyrics data
with open('az_lyrics.json', 'r') as file:
    lyrics_data = json.load(file)

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

heartbreak_keywords = {"break", "cry", "hurt", "tears", "pain", "goodbye", "leave", "gone", "lost", "sad", "alone", "depressed", "lonely"}
love_keywords = {"love", "heart", "kiss", "together", "darling", "forever", "sweet", "romance", "hold"}

# Initialize stopwords and sentiment analyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# Group songs by 1-year intervals
intervals = defaultdict(list)
for song in lyrics_data:
    try:
        year = int(song['year'])
    except:
        year = 0
    interval_start = (year // 2) * 2
    interval_end = interval_start + 1
    interval_key = f"{interval_start}-{interval_end}"
    
    # Clean and tokenize lyrics
    cleaned_lyrics = clean_text(song["lyrics"])
    tokens = [word for word in cleaned_lyrics.split() if word not in stop_words]
    intervals[interval_key].append(tokens)

# Analyze themes for each interval
for interval, lyrics_list in intervals.items():
    print(f"\nThemes from {interval}:")
    
    flattened_lyrics = [word for lyrics in lyrics_list for word in lyrics]
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary([flattened_lyrics])
    corpus = [dictionary.doc2bow(flattened_lyrics)]
    
    # Apply LDA
    lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    
    # Print topics
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)
    
    # Count dominant topic for each interval
    print(f"\nTopic distribution from {interval}:")
    corpus = [dictionary.doc2bow(text) for text in lyrics_list]
    topic_counts = [max(lda_model.get_document_topics(doc), key=lambda x: x[1])[0] for doc in corpus]
    topic_distribution = {i: topic_counts.count(i) for i in range(lda_model.num_topics)}
    print(topic_distribution)
    
    # Sentiment analysis and word counts
    word_counts = Counter()
    for lyrics in lyrics_list:
        sentence = ' '.join(lyrics)
        sentiment = sia.polarity_scores(sentence)['compound']
        
        if sentiment > 0:  # Positive sentiment
            word_counts.update(word for word in lyrics if word in love_keywords)
        elif sentiment < 0:  # Negative sentiment
            word_counts.update(word for word in lyrics if word in heartbreak_keywords)
    overall_words = Counter(word for lyrics in lyrics_list for word in lyrics)
    print(f"\nSentiment-adjusted word frequencies from {interval}:")
    print(word_counts.most_common(10))
    print(f"Most frequently used words from {interval}: ")
    print(overall_words.most_common(10))
    heartbreak_count = sum(word_counts[word] for word in heartbreak_keywords if word in word_counts)
    love_count = sum(word_counts[word] for word in love_keywords if word in word_counts)
    print(f"\nTheme frequencies from {interval}:")
    print(f"Heartbreak-related words: {heartbreak_count}")
    print(f"Love-related words: {love_count}")

# Plotting
intervals_sorted = intervals.keys()
heartbreak_trend = []
love_trend = []

for interval, lyrics_list in intervals.items():
    word_counts = Counter()
    for lyrics in lyrics_list:
        sentence = ' '.join(lyrics)
        sentiment = sia.polarity_scores(sentence)['compound']
        if sentiment > 0:  # Positive sentiment
            word_counts.update(word for word in lyrics if word in love_keywords)
        elif sentiment < 0:  # Negative sentiment
            word_counts.update(word for word in lyrics if word in heartbreak_keywords)
    heartbreak_count = sum(word_counts[word] for word in heartbreak_keywords if word in word_counts)
    love_count = sum(word_counts[word] for word in love_keywords if word in word_counts)
    heartbreak_trend.append(heartbreak_count)
    love_trend.append(love_count)

plt.plot(intervals_sorted, heartbreak_trend, label="Heartbreak", color="red")
plt.plot(intervals_sorted, love_trend, label="Love", color="green")
plt.xlabel("Time Interval")
plt.ylabel("Word Frequency")
plt.title("Sentiment-Adjusted Themes in Taylor Swift's Lyrics Over Time")
plt.legend()
plt.show()
