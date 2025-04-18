import json
import os
import sys
from typing import Dict, List, Optional, Union

from functions import (
    clean_text, remove_https, create_directory,
    tokenize_text, remove_stopwords, get_word_frequencies,
    simple_sentiment_analysis, extract_keywords, text_summarization,
    nltk_tokenize, advanced_sentiment_analysis, named_entity_recognition,
    save_nlp_results, analyze_text_document
)

RESULTS_DIR = "analysis_results"
create_directory(RESULTS_DIR)

FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, Request, Body, Query, Path, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not available. Running in demo mode.")
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    class TextRequest(BaseModel):
        text: str = Field(..., description="Text to analyze")

    class TokenizeRequest(BaseModel):
        text: str = Field(..., description="Text to tokenize")
        remove_stopwords: bool = Field(False, description="Whether to remove stopwords")

    class SentimentRequest(BaseModel):
        text: str = Field(..., description="Text to analyze sentiment")
        advanced: bool = Field(False, description="Whether to use advanced sentiment analysis")

    class KeywordsRequest(BaseModel):
        text: str = Field(..., description="Text to extract keywords from")
        top_n: int = Field(5, description="Number of top keywords to return")

    class SummarizeRequest(BaseModel):
        text: str = Field(..., description="Text to summarize")
        num_sentences: int = Field(3, description="Number of sentences in summary")

    class AnalyzeRequest(BaseModel):
        text: str = Field(..., description="Text to analyze")
        save_results: bool = Field(False, description="Whether to save results to file")


def demo_clean_text(text):
    print("\n===== CLEAN TEXT DEMO =====")
    cleaned_text = clean_text(text)
    print(f"Original: {text[:50]}...")
    print(f"Cleaned: {cleaned_text[:50]}...")
    return cleaned_text


def demo_tokenize(text):
    print("\n===== TOKENIZATION DEMO =====")
    tokens = tokenize_text(text)
    print(f"Tokens (first 10): {tokens[:10]}")
    print(f"Total tokens: {len(tokens)}")
    return tokens


def demo_remove_stopwords(tokens):
    print("\n===== STOPWORD REMOVAL DEMO =====")
    filtered_tokens = remove_stopwords(tokens)
    print(f"Tokens without stopwords (first 10): {filtered_tokens[:10]}")
    print(f"Tokens removed: {len(tokens) - len(filtered_tokens)}")
    return filtered_tokens


def demo_word_frequencies(tokens):
    print("\n===== WORD FREQUENCIES DEMO =====")
    word_freq = get_word_frequencies(tokens)
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 words: {top_words}")
    return word_freq


def demo_sentiment_analysis(text):
    print("\n===== SENTIMENT ANALYSIS DEMO =====")
    sentiment = simple_sentiment_analysis(text)
    print(f"Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")
    print(f"Positive words: {sentiment['positive_words']}, Negative words: {sentiment['negative_words']}")

    print("\nAdvanced sentiment analysis (fallback):")
    adv_sentiment = advanced_sentiment_analysis(text)
    print(f"Sentiment: {adv_sentiment['label']} (score: {adv_sentiment['score']:.2f})")
    return sentiment


def demo_keywords(text):
    print("\n===== KEYWORD EXTRACTION DEMO =====")
    keywords = extract_keywords(text, top_n=5)
    print(f"Top 5 keywords: {keywords}")
    return keywords


def demo_summarization(text):
    print("\n===== TEXT SUMMARIZATION DEMO =====")
    summary = text_summarization(text, num_sentences=2)
    print(f"Original text length: {len(text)} characters")
    print(f"Summary length: {len(summary)} characters")
    print(f"Reduction: {round((1 - len(summary) / len(text)) * 100, 2)}%")
    print(f"Summary: {summary}")
    return summary


def demo_named_entity_recognition(text):
    print("\n===== NAMED ENTITY RECOGNITION DEMO =====")
    entities = named_entity_recognition(text)
    print(f"Entities found: {entities}")
    return entities


def demo_comprehensive_analysis(text):
    print("\n===== COMPREHENSIVE ANALYSIS DEMO =====")
    print("Running comprehensive analysis...")

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"analysis_{timestamp}.json")
    results = analyze_text_document(text, output_file)
    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Results include: {', '.join(results.keys())}")
    print(f"\nWord count: {results['word_count']}")
    print(f"Unique words: {results['unique_words']}")
    print(f"Sentiment: {results['sentiment']['label']}")
    print(f"Top words: {list(results['top_words'].items())[:3]}...")
    print(f"Summary: {results['summary'][:100]}...")
    return results


def demo_check():
    print("\n===== ORIGINAL CHECK FUNCTION DEMO =====")
    data = {
        "name": "John Doe\n with extra spaces",
        "website": "https://example.com",
        "email": "john@example.com"
    }
    print(f"Original data: {data}")

    data["name"] = clean_text(data["name"])
    data["website"] = remove_https(data["website"])
    output_dir = "output"
    create_directory(output_dir)
    with open(f"{output_dir}/data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Processed data: {data}")
    print(f"Data saved to {output_dir}/data.json")
    return data


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="NLP API",
        description="API for Natural Language Processing tasks",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {
            "message": "Welcome to the NLP API",
            "endpoints": [
                "/api/clean-text",
                "/api/tokenize",
                "/api/sentiment",
                "/api/keywords",
                "/api/summarize",
                "/api/entities",
                "/api/analyze"
            ]
        }

    @app.post("/api/clean-text")
    async def api_clean_text(request: TextRequest):
        cleaned_text = clean_text(request.text)

        return {
            "original": request.text,
            "cleaned": cleaned_text
        }

    @app.post("/api/tokenize")
    async def api_tokenize(request: TokenizeRequest):
        tokens = tokenize_text(request.text)
        word_freq = get_word_frequencies(tokens)
        filtered_tokens = tokens
        if request.remove_stopwords:
            filtered_tokens = remove_stopwords(tokens)
        tokenized = nltk_tokenize(request.text)

        return {
            "tokens": tokens,
            "tokens_without_stopwords": filtered_tokens,
            "sentences": tokenized["sentences"],
            "word_count": len(tokens),
            "unique_words": len(word_freq),
            "top_words": dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
        }

    @app.post("/api/sentiment")
    async def api_sentiment(request: SentimentRequest):
        if request.advanced:
            sentiment = advanced_sentiment_analysis(request.text)
        else:
            sentiment = simple_sentiment_analysis(request.text)

        return sentiment

    @app.post("/api/keywords")
    async def api_keywords(request: KeywordsRequest):
        keywords = extract_keywords(request.text, top_n=request.top_n)

        return {
            "keywords": dict(keywords)
        }

    @app.post("/api/summarize")
    async def api_summarize(request: SummarizeRequest):
        summary = text_summarization(request.text, num_sentences=request.num_sentences)

        return {
            "original_length": len(request.text),
            "summary_length": len(summary),
            "reduction_percent": round((1 - len(summary) / len(request.text)) * 100, 2),
            "summary": summary
        }

    @app.post("/api/entities")
    async def api_entities(request: TextRequest):
        entities = named_entity_recognition(request.text)

        entity_types = {}
        for entity in entities:
            entity_type = entity["type"]
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity["text"])

        return {
            "entities": entities,
            "entity_types": entity_types,
            "entity_count": len(entities)
        }

    @app.post("/api/analyze")
    async def api_analyze(request: AnalyzeRequest):
        output_file = None
        if request.save_results:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(RESULTS_DIR, f"analysis_{timestamp}.json")
        results = analyze_text_document(request.text, output_file)
        results["entities"] = named_entity_recognition(request.text)

        return results

    @app.post("/check")
    async def check(request: Request):
        data = await request.json()
        if "name" in data:
            data["name"] = clean_text(data["name"])
        if "website" in data:
            data["website"] = remove_https(data["website"])
        output_dir = "output"
        create_directory(output_dir)
        with open(f"{output_dir}/data.json", "w") as f:
            json.dump(data, f, indent=2)

        return data

if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        print("\nNLP API is ready!")
        print("Run the following command to start the API server:")
        print("uvicorn main:app --reload")
        print("\nThen access the API documentation at http://localhost:8000/docs")
        print("\nOr run this script with --demo flag to see the demo:")
        print("python main.py --demo")
        if len(sys.argv) > 1 and sys.argv[1] == "--demo":
            run_demo = True
        else:
            sys.exit(0)
    else:
        run_demo = True

    if run_demo:
        def print_separator():
            print("\n" + "-" * 50 + "\n")

        sample_text = """
        Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction
        between computers and humans through natural language. The ultimate objective of NLP is to read, decipher,
        understand, and make sense of human language in a valuable way. NLP is used in many applications including
        chatbots, sentiment analysis, translation services, and text summarization. Companies like Google, Amazon,
        and Microsoft invest heavily in NLP research to improve their products and services.
        """

        text_to_analyze = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--demo" else sample_text

        print("\nNLP FUNCTIONS DEMO\n" + "=" * 20)
        print(f"Text to analyze: {text_to_analyze[:100]}...")
        print_separator()

        cleaned_text = demo_clean_text(text_to_analyze)
        tokens = demo_tokenize(text_to_analyze)
        filtered_tokens = demo_remove_stopwords(tokens)
        word_freq = demo_word_frequencies(filtered_tokens)
        sentiment = demo_sentiment_analysis(text_to_analyze)
        keywords = demo_keywords(text_to_analyze)
        summary = demo_summarization(text_to_analyze)
        entities = demo_named_entity_recognition(text_to_analyze)
        results = demo_comprehensive_analysis(text_to_analyze)
        demo_check()

        print_separator()
        print("Demo complete! All NLP functions have been demonstrated.")
        print("To use these functions in your own code, import them from functions.py")
