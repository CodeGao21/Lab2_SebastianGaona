import os
import requests
from dotenv import load_dotenv, find_dotenv
from autogen import AssistantAgent
import sys

# News API key
news_api_key = 'Mykey'

# Function to fetch news articles from News API
def fetch_news(api_key, query='technology', page_size=5):
    url = f'https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"News API request failed with status code {response.status_code}")
    return response.json()

# Fetch news articles
news_data = fetch_news(news_api_key)
articles = news_data.get('articles', [])

# Combine article content for the summarization task
news_content = "\n\n".join([f"{article['title']}\n{article['description']}" for article in articles])

# Redirect standard output to a file
original_stdout = sys.stdout
with open('punto2respuestas.txt', 'w') as f:
    sys.stdout = f
    
    gemini_api_key = 'MyKey'
    llm_config = { "model": "gemini-1.5-flash-latest", "api_key": gemini_api_key, "api_type": "google" }

    task = f'''
      Write a concise but engaging article summarizing the following news articles:
      {news_content}
    '''

    writer = AssistantAgent(
        name="Writer",
        system_message="You are a writer. You write engaging and concise "
            "articles (with title) on given topics. You must polish your "
            "writing based on the feedback you receive and give a refined "
            "version. Only return your final work without additional comments.",
        llm_config=llm_config,
    )

    critic = AssistantAgent(
        name="Critic",
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        llm_config=llm_config,
        system_message="You are a critic. You review the work of "
                    "the writer and provide constructive "
                    "feedback to help improve the quality of the content.",
    )

    SEO_reviewer = AssistantAgent(
        name="SEO Reviewer",
        llm_config=llm_config,
        system_message="You are an SEO reviewer, known for "
            "your ability to optimize content for search engines, "
            "ensuring that it ranks well and attracts organic traffic. "
            "Make sure your suggestion is concise (within 3 bullet points), "
            "concrete and to the point. "
            "Begin the review by stating your role.",
    )

    legal_reviewer = AssistantAgent(
        name="Legal Reviewer",
        llm_config=llm_config,
        system_message="You are a legal reviewer, known for "
            "your ability to ensure that content is legally compliant "
            "and free from any potential legal issues. "
            "Make sure your suggestion is concise (within 3 bullet points), "
            "concrete and to the point. "
            "Begin the review by stating your role.",
    )

    ethics_reviewer = AssistantAgent(
        name="Ethics Reviewer",
        llm_config=llm_config,
        system_message="You are an ethics reviewer, known for "
            "your ability to ensure that content is ethically sound "
            "and free from any potential ethical issues. "
            "Make sure your suggestion is concise (within 3 bullet points), "
            "concrete and to the point. "
            "Begin the review by stating your role. ",
    )

    meta_reviewer = AssistantAgent(
        name="Meta Reviewer",
        llm_config=llm_config,
        system_message="You are a meta reviewer, you aggregate and review "
        "the work of other reviewers and give a final suggestion on the content.",
    )

    def reflection_message(recipient, messages, sender, config):
        return f'''Review the following content.
                \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

    review_chats = [
      {
        "recipient": SEO_reviewer,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_args": {
          "summary_prompt" :
            "Return review into as JSON object only: {'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",
        },
        "max_turns": 1
      },
      {
        "recipient": legal_reviewer, "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_args": {
          "summary_prompt" :
            "Return review into as JSON object only: {'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",
        },
        "max_turns": 1
      },
      {
        "recipient": ethics_reviewer, "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_args": {
          "summary_prompt" :
            "Return review into as JSON object only: {'reviewer': '', 'review': ''}.  Here Reviewer should be your role",
        },
        "max_turns": 1
      },
      {
        "recipient": meta_reviewer,
        "message": "Aggregate feedback from all reviewers and give final suggestions on the writing.",
        "max_turns": 1
      },
    ]

    critic.register_nested_chats(
        review_chats,
        trigger=writer,
    )

    res = critic.initiate_chat(
        recipient=writer,
        message=task,
        max_turns=2,
        summary_method="last_msg"
    )

    print(res.summary)

# Restore the standard output to the console
sys.stdout = original_stdout

print("Output has been written to punto2respuestas.txt")


