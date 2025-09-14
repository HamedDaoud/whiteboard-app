import sys
from pathlib import Path
import logging

# Add the src directory to the path so we can import our modules
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from whiteboard.content_generator import generate_content_for_topic, fetch_lessons

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Check if topic is provided as command line argument
    if len(sys.argv) < 2:
        logger.error("Please provide a topic as a command line argument (e.g., python test_content_generation.py 'Linear Algebra')")
        sys.exit(1)
    
    test_topic = sys.argv[1]
    
    logger.info(f"Testing content generation for topic: {test_topic}")
    logger.info("This may take a few moments to query the Hugging Face API...")
    
    try:
        # Generate and save content
        content = generate_content_for_topic(test_topic, k=4)
        
        print("\n" + "="*50)
        print("GENERATION TEST RESULTS")
        print("="*50)
        
        print(f"\nTOPIC: {content.topic}")
        
        print(f"\nLESSON:")
        print(content.lesson[:500] + "..." if len(content.lesson) > 500 else content.lesson)
        
        print(f"\nQUIZ:")
        quiz = content.quiz.get('quiz', [])
        if not quiz:
            logger.warning("No quiz questions generated.")
        for i, q in enumerate(quiz, 1):
            print(f"Q{i}: {q.get('question')}")
            if q.get('options'):
                for opt_key, opt_value in q.get('options', {}).items():
                    print(f"    {opt_key}) {opt_value}")
            print(f"    Correct: {q.get('correct_answer')}\n")
            
        # Verify the save by fetching lessons
        print("\n" + "="*50)
        print("VERIFYING SUPABASE STORAGE")
        print("="*50)
        lessons = fetch_lessons()
        if lessons:
            print(f"Fetched {len(lessons)} lessons from Supabase:")
            for i, lesson in enumerate(lessons, 1):
                print(f"Lesson {i}: Topic = {lesson.get('topic')}, Created At = {lesson.get('created_at')}")
        else:
            print("No lessons found in Supabase. Check if the save operation succeeded.")
            
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")
        if "503" in str(e) or "rate limit" in str(e).lower():
            logger.error("Hugging Face API rate limit exceeded or service unavailable. Try again later.")
        elif "401" in str(e) or "authentication" in str(e).lower():
            logger.error("Invalid Hugging Face API token. Verify the token in content_generator.py.")
        # Print the full error for debugging
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()