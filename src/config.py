# Constants
CHROMA_PATH = "../chroma"
DATA_PATH = '../data'
SESSION_FILE = "session.json"

API_KEYS = {
    "llama": "GROQ API KEY",
    "mixtral": "GROQ API KEY",
    "openai": "OpenAI API KEY",
    "google": "Google AI Studio API KEY"
}

INITIAL_PROMPT_TEMPLATE = """
--------------------
DOCUMENTEN: 
{context}
--------------------

AFBEELDINGSBESCHRIJVING:
{image_description}
--------------------

VRAAG: {question}
--------------------

INSTRUCTIES:
Beantwoord de VRAAG gegeven de DOCUMENTEN en AFBEELDINGSBESCHRIJVING.
Houd je antwoorden gegrond in de feiten in de DOCUMENTEN.
Je mag ook eventuele links meegegeven die leiden naar de webpagina waar het antwoord te vinden is.
"""

SESSION_PROMPT_TEMPLATE = """
--------------------
DOCUMENTEN: 
{context}
--------------------

CHATGESCHIEDENIS: 
{history}
--------------------

AFBEELDINGSBESCHRIJVING:
{image_description}
--------------------

VRAAG: {question}
--------------------

INSTRUCTIES:
Beantwoord de VRAAG gegeven de DOCUMENTEN, CHATGESCHIEDENIS en AFBEELDINGSBESCHRIJVING.
Houd je antwoord gegrond in de feiten in de DOCUMENTEN en CHATGESCHIEDENIS.
Je mag ook eventuele links meegegeven die leiden naar de webpagina waar het antwoord te vinden is.
"""

SUMMARIZE_PROMPT_TEMPLATE = """
--------------------
CHATGESCHIEDENIS:
{history}
--------------------

AFBEELDINGSBESCHRIJVING:
{image_description}
--------------------

HUIDIGE VRAAG: {query}
--------------------

INSTRUCTIES:
Geef een samenvatting van de huidige VRAAG, de AFBEELDINGSBESCHRIJVING en de vragen in de CHATGESCHIEDENIS als een zoekterm.
Geef alleen een samenvatting van de huidige VRAAG als de CHATGESCHIEDENIS en de AFBEELDINGSBESCHRIJVING niet gerelateerd zijn aan de HUIDIGE VRAAG.
Geef ALLEEN deze geparafraseerde zoekterm terug.
"""