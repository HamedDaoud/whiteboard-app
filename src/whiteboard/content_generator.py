from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
from huggingface_hub import InferenceClient
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the retrieval function to get chunks for a topic
from .retrieval import get_chunks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Configuration & Model Setup ----------------

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Model supports conversational task
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

client = None

try:
    logger.info(f"Initializing Hugging Face Inference Client for model: {MODEL_NAME}...")
    client = InferenceClient(model=MODEL_NAME, token=HF_API_TOKEN)
    logger.info("Inference client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize inference client for '{MODEL_NAME}': {e}")
    logger.error("The content generator will not function. Please check:")
    logger.error("  1. Your internet connection.")
    logger.error("  2. The validity of the Hugging Face API token.")
    logger.error("  3. Whether the model is supported by the Hugging Face Inference API.")
    logger.error("  Visit https://huggingface.co/docs/inference-api for supported models.")

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_KEY is not set in environment variables.")
    raise ValueError("Supabase configuration is missing. Please set SUPABASE_URL and SUPABASE_KEY in .env.")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    raise

# ---------------- Prompt Templates ----------------

_PROMPT_TEMPLATE_LESSON = """
Synthesize the information from the context below into a detailed, clear, and engaging lesson summary of 300-500 words.
The lesson must include:
1. **Introduction**: A brief overview of the topic (3-4 sentences) that captures its importance and relevance.
2. **Main Points**: 3-5 key points explained logically in bullet points, using clear, simple language suitable for a student. Each point should be elaborated with sufficient detail to enhance understanding.
3. **Example**: A practical, relatable example that illustrates one or more of the main points, with enough context to make it engaging and clear.
4. **Sources**: List the sources of the information, including any available metadata (e.g., title, URL, section). If no metadata is provided, use a brief snippet (first 50 characters) of the chunk text as an identifier, followed by: "(No metadata provided)."
Do not invent information outside the provided context. Format the output clearly with markdown headers and bullet points for readability.

CONTEXT:
{context}

SOURCES:
{sources}

LESSON:
"""

_PROMPT_TEMPLATE_QUIZ = """
Based ONLY on the following lesson, create exactly 3 multiple-choice questions and one open-ended question.

Rules:
- Each question MUST end with "|||"
- Use this format for MCQs:
  QUESTION: ...
  OPTIONS: A) ... B) ... C) ... D) ...
  ANSWER: ...
- Use this format for open-ended:
  QUESTION: ...
  ANSWER: ...
- Do NOT add extra text, JSON, or markdown.

Lesson:
{lesson}
"""

# ---------------- Core Generation Functions ----------------

def _generate_text(prompt: str, max_new_tokens: int = 1024) -> str:
    if client is None:
        raise RuntimeError("Inference client is not initialized.")
    
    try:
        # Use chat_completion for conversational task
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stream=False,
        )
        # Extract the generated text from the assistant's response
        output = response.choices[0].message.content
        return output.strip()
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        if "503" in str(e) or "rate limit" in str(e).lower():
            logger.error("Hugging Face API rate limit exceeded or service unavailable. Try again later.")
        elif "401" in str(e) or "authentication" in str(e).lower():
            logger.error("Invalid Hugging Face API token. Verify the token.")
        elif "model" in str(e).lower() or "provider" in str(e).lower():
            logger.error(f"Model '{MODEL_NAME}' may not be supported for the requested task. Check https://huggingface.co/docs/inference-api.")
        raise

def generate_lesson_from_chunks(retrieved_chunks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    # Log the structure of retrieved_chunks for debugging
    logger.info(f"Retrieved chunks structure: {[chunk.keys() for chunk in retrieved_chunks]}")
    logger.info(f"Source sample: {[chunk.get('source', 'No source') for chunk in retrieved_chunks]}")
    
    # Extract source information from chunks, handling missing or incomplete source data
    sources = "\n".join([
        (
            f"- {chunk.get('source', {}).get('title', 'Unknown Title')} "
            f"({chunk.get('source', {}).get('url', 'No URL provided')})"
            f"{', Section: ' + chunk['source']['section'] if chunk.get('source', {}).get('section') else ''}"
            if chunk.get('source', {}) and any(chunk.get('source', {}).get(key) for key in ['title', 'url', 'section'])
            else f"- {chunk['text'][:50].strip()}... (No source metadata provided)"
        )
        for chunk in retrieved_chunks
    ])
    # Truncate context to avoid exceeding API limits
    if len(context) > 4000:
        logger.warning("Context is very long. Truncating...")
        context = context[:4000]
    
    prompt = _PROMPT_TEMPLATE_LESSON.format(context=context, sources=sources)
    lesson_text = _generate_text(prompt, max_new_tokens=1500)
    return lesson_text

def generate_quiz_from_lesson(lesson_text: str) -> Dict[str, Any]:
    prompt = _PROMPT_TEMPLATE_QUIZ.format(lesson=lesson_text)
    quiz_text = _generate_text(prompt, max_new_tokens=512)
    logger.info(f"Raw quiz output from model: {quiz_text}")

    questions = []
    
    # Split by QUESTION: to separate each question block
    question_blocks = re.split(r'(?=QUESTION:)', quiz_text)
    
    for block in question_blocks:
        block = block.strip()
        if not block or not block.startswith("QUESTION:"):
            continue
            
        try:
            # Extract question
            question_match = re.search(r'QUESTION:\s*(.*?)(?=OPTIONS:|ANSWER:|$)', block, re.DOTALL)
            if not question_match:
                continue
            question = question_match.group(1).strip()
            
            # Check if it's multiple choice or open-ended
            if "OPTIONS:" in block:
                # Multiple choice question
                options_match = re.search(r'OPTIONS:\s*(.*?)(?=ANSWER:|$)', block, re.DOTALL)
                answer_match = re.search(r'ANSWER:\s*(.*?)$', block, re.DOTALL)
                
                if not options_match or not answer_match:
                    continue
                    
                options_text = options_match.group(1).strip()
                correct_answer = answer_match.group(1).strip()
                
                # Parse options using regex
                options = {}
                option_pattern = r'([A-D])\)\s*(.*?)(?=\s+[A-D]\)|$)'
                for match in re.finditer(option_pattern, options_text):
                    option_letter = match.group(1)
                    option_text = match.group(2).strip()
                    options[option_letter] = option_text
                
                if question and options and correct_answer:
                    questions.append({
                        "type": "multiple_choice",
                        "question": question,
                        "options": options,
                        "correct_answer": correct_answer
                    })
                    
            else:
                # Open-ended question
                answer_match = re.search(r'ANSWER:\s*(.*?)$', block, re.DOTALL)
                if answer_match:
                    correct_answer = answer_match.group(1).strip()
                    questions.append({
                        "type": "open_ended",
                        "question": question,
                        "options": None,
                        "correct_answer": correct_answer
                    })
                    
        except Exception as e:
            logger.warning(f"Failed to parse block: {block[:100]}... ({e})")
            continue

    return {"quiz": questions}

# ---------------- Supabase Functions ----------------

def save_content(content: GeneratedContent) -> bool:
    """
    Save generated content to the Supabase lessons table.
    Returns True if successful, False otherwise.
    """
    try:
        # Validate required fields
        if not content.topic or not content.lesson or not content.quiz:
            logger.error("Missing required fields in content object.")
            logger.error(f"Topic: {content.topic}, Lesson: {len(content.lesson) if content.lesson else None}, Quiz: {content.quiz}")
            return False

        data = {
            "topic": content.topic,
            "lesson": content.lesson,
            "quiz": content.quiz,
            "retrieved_chunks": content.retrieved_chunks,
        }
        logger.info(f"Attempting to insert data into lessons table: {data}")
        
        response = supabase.table("lessons").insert(data).execute()
        logger.info(f"Insert successful: {response.data}")
        return True
    except Exception as e:
        logger.error(f"Failed to save content to Supabase: {e}")
        logger.error("Check Supabase URL, key, table permissions, and schema.")
        return False

def fetch_lessons() -> List[Dict[str, Any]]:
    """
    Fetch all lessons from the Supabase lessons table, ordered by created_at descending.
    """
    try:
        response = supabase.table("lessons").select("*").order("created_at", desc=True).execute()
        logger.info(f"Fetched {len(response.data)} lessons from Supabase.")
        return response.data
    except Exception as e:
        logger.error(f"Failed to fetch lessons from Supabase: {e}")
        return []

# ---------------- Main Public Function ----------------

@dataclass
class GeneratedContent:
    topic: str
    lesson: str
    quiz: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]

def generate_content_for_topic(topic: str, query: str | None = None, k: int = 6) -> GeneratedContent:
    logger.info(f"Generating content for topic: {topic}")
    retrieved_chunks = get_chunks(topic, query=query, k=k)
    logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")
    lesson_text = generate_lesson_from_chunks(retrieved_chunks)
    logger.info("Lesson generated successfully.")
    quiz_data = generate_quiz_from_lesson(lesson_text)
    logger.info(f"Quiz generated with {len(quiz_data.get('quiz', []))} questions.")
    content = GeneratedContent(
        topic=topic,
        lesson=lesson_text,
        quiz=quiz_data,
        retrieved_chunks=retrieved_chunks
    )
    # Save to Supabase
    if save_content(content):
        logger.info(f"Content for topic '{topic}' saved to Supabase.")
    else:
        logger.error(f"Failed to save content for topic '{topic}' to Supabase.")
    return content