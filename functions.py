import os.path
import re
import string
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union

# Uncomment these imports if you have the libraries installed
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from transformers import pipeline


# Basic text cleaning functions
def clean_text(text: str) -> str:
    """Clean text by removing whitespace and converting to uppercase.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    clean = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("  ", " ")
    clean = clean.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
    clean = clean.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
    clean = clean.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
    clean = clean.upper()
    return clean


def remove_https(url: str) -> str:
    return url.replace("https://", "-").replace("http://", "-")


def create_directory(path="os.getcwd()"):
    if not os.path.exists(path):
        os.makedirs('{}/'.format(path))
        print("Directory created: {}".format(path))
        return True
    else:
        print("Directory already exists: {}".format(path))
        return False


def tokenize_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens


def remove_stopwords(tokens: List[str], custom_stopwords: Optional[List[str]] = None) -> List[str]:
    stopwords_list = [
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    ]

    if custom_stopwords:
        stopwords_list.extend(custom_stopwords)
    filtered_tokens = [token for token in tokens if token not in stopwords_list]
    return filtered_tokens


def get_word_frequencies(tokens: List[str]) -> Dict[str, int]:
    return dict(Counter(tokens))


def simple_sentiment_analysis(text: str) -> Dict[str, Union[float, str]]:
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'terrific',
        'outstanding', 'superb', 'brilliant', 'awesome', 'happy', 'love', 'best', 'positive',
        'beautiful', 'perfect', 'nice', 'enjoy', 'pleased', 'joy', 'success', 'win', 'winning'
    ]

    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing', 'negative',
        'worst', 'hate', 'dislike', 'sad', 'angry', 'upset', 'unhappy', 'fail', 'failure',
        'problem', 'trouble', 'difficult', 'wrong', 'error', 'mistake', 'broken', 'damage'
    ]
    text = text.lower()
    tokens = tokenize_text(text)
    positive_count = sum(1 for token in tokens if token in positive_words)
    negative_count = sum(1 for token in tokens if token in negative_words)
    total_count = positive_count + negative_count
    if total_count == 0:
        score = 0.0
    else:
        score = (positive_count - negative_count) / total_count
    if score > 0.1:
        label = 'positive'
    elif score < -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    return {
        'score': score,
        'label': label,
        'positive_words': positive_count,
        'negative_words': negative_count
    }


def extract_keywords(text: str, top_n: int = 5) -> List[Tuple[str, int]]:
    tokens = tokenize_text(text)
    filtered_tokens = remove_stopwords(tokens)
    word_freq = get_word_frequencies(filtered_tokens)
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return keywords


def text_summarization(text: str, num_sentences: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= num_sentences:
        return text
    word_freq = get_word_frequencies(tokenize_text(text))
    sentence_scores = []
    for sentence in sentences:
        score = sum(word_freq.get(word, 0) for word in tokenize_text(sentence))
        score = score / (len(tokenize_text(sentence)) + 1)
        sentence_scores.append((sentence, score))
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    summary_sentences = []
    for sentence, _ in top_sentences:
        original_index = sentences.index(sentence)
        summary_sentences.append((original_index, sentence))
    summary_sentences.sort()
    summary = ' '.join(sentence for _, sentence in summary_sentences)
    return summary


def nltk_tokenize(text: str) -> Dict[str, List[str]]:
    # Uncomment to use NLTK
    # word_tokens = word_tokenize(text)
    # sentence_tokens = sent_tokenize(text)
    # return {'words': word_tokens, 'sentences': sentence_tokens}

    # Fallback implementation
    words = tokenize_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return {'words': words, 'sentences': sentences}


def advanced_sentiment_analysis(text: str) -> Dict:
    # Uncomment to use transformers
    # sentiment_analyzer = pipeline('sentiment-analysis')
    # result = sentiment_analyzer(text)
    # return result[0]

    return simple_sentiment_analysis(text)


def named_entity_recognition(text: str) -> List[Dict]:
    # Uncomment to use spaCy
    # nlp = spacy.load('en_core_web_sm')
    # doc = nlp(text)
    # entities = [{'text': ent.text, 'type': ent.label_} for ent in doc.ents]
    # return entities

    # Simple fallback (very limited)
    entities = []
    for match in re.finditer(r'\b[A-Z][a-z]+\b', text):
        entities.append({'text': match.group(), 'type': 'UNKNOWN'})
    return entities


def save_nlp_results(results: Dict, output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def analyze_text_document(text: str, output_file: Optional[str] = None) -> Dict:
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords(tokens)
    word_freq = get_word_frequencies(filtered_tokens)
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    sentiment = simple_sentiment_analysis(cleaned_text)
    keywords = extract_keywords(cleaned_text)
    summary = text_summarization(text)

    results = {
        'word_count': len(tokens),
        'unique_words': len(word_freq),
        'top_words': dict(top_words),
        'sentiment': sentiment,
        'keywords': dict(keywords),
        'summary': summary
    }
    if output_file:
        save_nlp_results(results, output_file)
    return results

