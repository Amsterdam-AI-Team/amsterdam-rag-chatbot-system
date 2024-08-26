
# Amsterdam RAG Chatbot System

Retrieval Augmented Generation (RAG) Chatbot System designed for the Municipality of Amsterdam. This system answers questions regarding municipal information on the website, supports both textual and image prompts, and leverages multiple Large Language Models (LLMs) for generation and text-to-speech streaming.

## Background

The Amsterdam RAG Chatbot System is a project developed to assist citizens and visitors of Amsterdam in easily accessing information related to the municipality. By integrating advanced AI technologies, such as Retrieval Augmented Generation (RAG), the system provides accurate and context-aware responses. The chatbot supports multiple input types, including text and images, and offers flexibility in response generation through different LLMs.

## Folder Structure

* [`data`](./data): Sample data for demo purposes.
* [`src`](./src): All source code files specific to this project.

## Installation 

1) Clone this repository:

```bash
git clone https://github.com/Amsterdam-AI-Team/amsterdam-rag-chatbot-system.git
```

2) Install all dependencies:

```bash
pip install -r requirements.txt
```

The code has been tested with Python x.x on Linux/MacOS/Windows.

## Usage

### Step 1: Populate the Database

Before running the chatbot, you need to populate the database with chunked documents. This is done by running the following script:

```bash
python populate_database.py
```

This script will process and store the documents in the database, making them available for retrieval during chatbot interactions.

### Step 2: Run the Chatbot

Once the database is populated, you can run the chatbot locally by executing the following command:

```bash
python app.py
```

This will start the chatbot on your localhost, allowing you to interact with it via a web interface.

## Contributing

We welcome contributions! Feel free to [open an issue](https://github.com/Amsterdam-AI-Team/amsterdam-rag-chatbot-system/issues), submit a [pull request](https://github.com/Amsterdam-AI-Team/amsterdam-rag-chatbot-system/pulls), or [contact us](https://amsterdamintelligence.com/contact/) directly.

## Acknowledgements

This repository was created by [Amsterdam Intelligence](https://amsterdamintelligence.com/) for the City of Amsterdam.

Optional: add citation or references here.

## License 

This project is licensed under the terms of the European Union Public License 1.2 (EUPL-1.2).
