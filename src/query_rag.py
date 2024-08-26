import argparse
import base64
import json
import os

import google.generativeai as GoogleGenAI
import requests
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from openai import OpenAI

import config as cfg
from helpers.embedding_helpers import OpenAIEmbeddingFunction

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class QueryProcessor:
    """
    A class to process queries using various language models, including summarizing queries,
    searching a database, creating prompts, invoking models, and displaying results.

    Attributes:
        query_text (str): The query text to be processed.
        model_name (str): The name of the language model to be used.
        chat_history (list, optional): The history of the chat session.
    """

    def __init__(self, query_text, model_name, base64_image=None, chat_history=None):
        """
        Initialize the QueryProcessor with the query text, model name, API key, and chat history.

        Args:
            query_text (str): The query text to process.
            model_name (str): The name of the model to use.
            base64_iamge (base64, optional): The image as base64.
            chat_history (list, optional): Previous chat history as a list of dictionaries. Defaults to an empty list.
        """
        self.query_text = query_text
        self.model_name = model_name
        self.base64_image = base64_image
        self.chat_history = chat_history or []

    def process_query(self):
        """
        Process the query by summarizing it if there is chat history,
        searching the database, creating the prompt, invoking the model,
        and displaying the response.
        """
        image_description = None
        if self.base64_image:
            image_description = self.process_image(self.base64_image)
            print(image_description)

        summarized_query = self.summarize_query(image_description)
        context_text, sources = self.search_db(summarized_query)
        prompt = self.create_prompt(context_text, image_description)
        response_text = self.invoke_model(prompt)
        self.chat_history.append({"vraag": self.query_text, "antwoord": response_text})
        self.format_response(response_text, sources)

    def process_image(self, base64_image):
        """
        Generate discription of image.

        Args:
            base64_image (base64): The image as base64.

        Returns:
            json: A json containing:
                - content (str): Description of image.
                - total_tokens (int): total tokens used to process image.
        """
        client = OpenAI(api_key=cfg.API_KEYS["openai"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Wat staat er op deze afbeelding?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens = 300
        )
        return response.choices[0].message.content

    def summarize_query(self, image_description=None):
        """
        Summarize the current query along with the chat history and optional image description.

        Args:
            image_description (str, optional): Description of the image to include in summarization.

        Returns:
            str: The summarized query.
        """
        history_text = "\n".join(
            [f"Vraag: {entry['vraag']}\nAntwoord: {entry['antwoord']}" for entry in self.chat_history]
        )
        summarize_prompt = ChatPromptTemplate.from_template(cfg.SUMMARIZE_PROMPT_TEMPLATE)
        prompt = summarize_prompt.format(
            history=history_text, 
            query=self.query_text, 
            image_description=image_description or ""
        )
        summarized_query = self.invoke_model(prompt, summarize=True)
        
        return summarized_query

    def search_db(self, summarized_query):
        """
        Search the Chroma database using the summarized query,
        combining and deduplicating results.

        Args:
            summarized_query (str): The summarized query text.

        Returns:
            tuple: A tuple containing:
                - context_text (str): Combined context text from the search results.
                - sources (list): List of sources used in the search.
        """
        db = Chroma(persist_directory=cfg.CHROMA_PATH, embedding_function=OpenAIEmbeddingFunction())

        summarized_results = (
            db.similarity_search_with_score(summarized_query, k=10) if summarized_query else []
        )

        combined_results = {
            doc.metadata.get("source"): doc for doc, _ in summarized_results
        }
        sources = combined_results.keys()
        context_text = "\n\n---\n\n".join([open(source, "r").read() for source in sources])
        
        return context_text, sources

    def create_prompt(self, context_text, image_description=None):
        """
        Create a prompt based on whether chat history is present or not, including image description if provided.

        Args:
            context_text (str): The text context generated from the search results.
            image_description (str, optional): Description of the image to include in the prompt.

        Returns:
            str: The formatted prompt.
        """
        if self.chat_history:
            history_text = "\n".join(
                [f"Vraag: {entry['vraag']}\nAntwoord: {entry['antwoord']}" for entry in self.chat_history]
            )
            prompt_template = ChatPromptTemplate.from_template(cfg.SESSION_PROMPT_TEMPLATE)
            prompt = prompt_template.format(
                context=context_text, 
                question=self.query_text, 
                history=history_text, 
                image_description=image_description or ""
            )
        else:
            prompt_template = ChatPromptTemplate.from_template(cfg.INITIAL_PROMPT_TEMPLATE)
            prompt = prompt_template.format(
                context=context_text, 
                question=self.query_text, 
                image_description=image_description or ""
            )
        
        return prompt


    def invoke_model(self, prompt, summarize=False):
        """
        Invoke the language model to generate a response based on the provided prompt.

        Args:
            prompt (str): The formatted prompt to be sent to the language model.
            summarize (bool, optional): If True, use a summarizing system message. Defaults to False.

        Returns:
            str: The generated response from the language model.
        """
        system_content = "Je bent een behulpzame assistent" if summarize else "Je bent een behulpzame ambtenaar."
        
        if self.model_name == "Llama 3 (70b)":
            client = Groq(api_key=cfg.API_KEYS["llama"])
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
            )

        elif self.model_name == "Mixtral (8x7b)":
            client = Groq(api_key=cfg.API_KEYS["mixtral"])
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
            )

        elif self.model_name == "ChatGPT 4o mini":
            client = OpenAI(api_key=cfg.API_KEYS["openai"])
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
            )
            
        elif self.model_name == "Gemini Pro":
            GoogleGenAI.configure(api_key=cfg.API_KEYS["google"])
            model = GoogleGenAI.GenerativeModel(model_name="gemini-1.5-pro")
            response_text = model.generate_content(prompt).text
            return response_text
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return completion.choices[0].message.content

    def format_response(self, response_text, sources):
        """
        format the response text and the sources used to generate it.

        Args:
            response_text (str): The text generated by the language model.
            sources (list): List of sources used in generating the response.

        Returns:
            str: The formatted response string with sources included.
        """
        formatted_response = f"Response: {response_text}\nSources: {', '.join(sources)}"
        return formatted_response

def load_session():
    """
    Load session data from the session file if it exists.

    Returns:
        list: A list containing the chat history loaded from the session file. 
              If the file does not exist, an empty list is returned.
    """
    if os.path.exists(cfg.SESSION_FILE):
        with open(cfg.SESSION_FILE, "r") as f:
            return json.load(f).get("chat_history", [])
    return []

def save_session(chat_history):
    """
    Save the current chat history to the session file.

    Args:
        chat_history (list): A list containing the chat history to be saved.
    """
    with open(cfg.SESSION_FILE, "w") as f:
        json.dump({"chat_history": chat_history}, f)

def main():
    """
    Main function to parse arguments, load session data, process the query, 
    and save the session data.
    """
    parser = argparse.ArgumentParser(description="Process a query using a specified language model.")
    parser.add_argument("query_text", type=str, help="The query text to process.")
    parser.add_argument("--model", type=str, choices=["Llama 3 (70b)", "ChatGPT 4o mini", "Gemini Pro", "Mixtral (8x7b)"], 
                        default="ChatGPT 4o mini", help="The model to use for processing.")
    parser.add_argument("--session", action="store_true", help="Continue with the previous session if available.")
    
    args = parser.parse_args()

    chat_history = load_session() if args.session else []

    processor = QueryProcessor(args.query_text, args.model, chat_history)
    processor.process_query()

    save_session(processor.chat_history)

if __name__ == "__main__":
    main()