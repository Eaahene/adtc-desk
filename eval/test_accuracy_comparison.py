"""
Test accuracy on 1.5B model with arc_easy subset.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama

MODELS_DIR = Path(__file__).parent.parent / "models"

# Load arc_easy questions (first 20 for quick test)
ARC_EASY_PATH = Path(__file__).parent / "arc_easy_sample.json"

def load_arc_easy():
    """Load or create arc_easy sample questions."""
    if ARC_EASY_PATH.exists():
        with open(ARC_EASY_PATH) as f:
            return json.load(f)
    
    # Create sample questions
    questions = [
        {
            "question": "Which of the following is an example of a physical change?",
            "choices": ["A. Burning wood", "B. Melting ice", "C. Rusting iron", "D. Cooking an egg"],
            "answer": "B"
        },
        {
            "question": "Which planet is closest to the Sun?",
            "choices": ["A. Venus", "B. Mercury", "C. Mars", "D. Earth"],
            "answer": "B"
        },
        {
            "question": "What is the process by which plants make their own food?",
            "choices": ["A. Respiration", "B. Photosynthesis", "C. Fermentation", "D. Digestion"],
            "answer": "B"
        },
        {
            "question": "Which of these is a renewable resource?",
            "choices": ["A. Coal", "B. Natural gas", "C. Solar energy", "D. Oil"],
            "answer": "C"
        },
        {
            "question": "What force keeps planets in orbit around the Sun?",
            "choices": ["A. Magnetic force", "B. Electric force", "C. Gravity", "D. Friction"],
            "answer": "C"
        },
        {
            "question": "Which state of matter has a fixed shape and volume?",
            "choices": ["A. Gas", "B. Liquid", "C. Plasma", "D. Solid"],
            "answer": "D"
        },
        {
            "question": "What is the main function of the heart?",
            "choices": ["A. To filter blood", "B. To pump blood", "C. To produce blood", "D. To store blood"],
            "answer": "B"
        },
        {
            "question": "Which of the following is an acid?",
            "choices": ["A. Sodium hydroxide", "B. Hydrochloric acid", "C. Baking soda", "D. Table salt"],
            "answer": "B"
        },
        {
            "question": "What type of energy does the Sun primarily emit?",
            "choices": ["A. Chemical energy", "B. Nuclear energy", "C. Light energy", "D. Electrical energy"],
            "answer": "C"
        },
        {
            "question": "Which organ in the human body is responsible for thinking?",
            "choices": ["A. Heart", "B. Liver", "C. Brain", "D. Kidneys"],
            "answer": "C"
        },
        {
            "question": "What is the boiling point of water at sea level?",
            "choices": ["A. 50°C", "B. 75°C", "C. 100°C", "D. 150°C"],
            "answer": "C"
        },
        {
            "question": "Which of these is a solid?",
            "choices": ["A. Oxygen", "B. Water", "C. Iron", "D. Helium"],
            "answer": "C"
        },
        {
            "question": "What causes the tides in the ocean?",
            "choices": ["A. Wind", "B. Earthquakes", "C. Moon's gravity", "D. Water temperature"],
            "answer": "C"
        },
        {
            "question": "Which gas do humans breathe in to survive?",
            "choices": ["A. Carbon dioxide", "B. Nitrogen", "C. Oxygen", "D. Hydrogen"],
            "answer": "C"
        },
        {
            "question": "What is the closest star to Earth?",
            "choices": ["A. Sirius", "B. The Sun", "C. Alpha Centauri", "D. Polaris"],
            "answer": "B"
        },
        {
            "question": "Which of these materials is a conductor of electricity?",
            "choices": ["A. Rubber", "B. Plastic", "C. Copper", "D. Wood"],
            "answer": "C"
        },
        {
            "question": "What is the process of water turning into vapor called?",
            "choices": ["A. Condensation", "B. Freezing", "C. Evaporation", "D. Precipitation"],
            "answer": "C"
        },
        {
            "question": "Which layer of the Earth is the thickest?",
            "choices": ["A. Crust", "B. Mantle", "C. Outer core", "D. Inner core"],
            "answer": "B"
        },
        {
            "question": "What type of rock is formed by volcanic activity?",
            "choices": ["A. Sedimentary", "B. Metamorphic", "C. Igneous", "D. Limestone"],
            "answer": "C"
        },
        {
            "question": "Which of these is a habitat for fish?",
            "choices": ["A. Desert", "B. Forest", "C. Ocean", "D. Mountain"],
            "answer": "C"
        }
    ]
    
    with open(ARC_EASY_PATH, "w") as f:
        json.dump(questions, f, indent=2)
    
    return questions

def test_accuracy(model_path, questions):
    """Test accuracy on arc_easy questions."""
    print(f"Loading model: {model_path.name}")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=6,
        n_batch=64,
        verbose=False,
        use_mmap=True,
        use_mlock=True,
        n_gpu_layers=0,
    )
    
    correct = 0
    total = len(questions)
    
    for i, q in enumerate(questions, 1):
        prompt = f"Question: {q['question']}\n\nChoices:\n"
        for choice in q['choices']:
            prompt += f"{choice}\n"
        prompt += "\nAnswer (just the letter):"
        
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1,
        )
        
        response = result["choices"][0]["message"]["content"].strip().upper()
        predicted = response[0] if response else "X"
        expected = q['answer']
        
        if predicted == expected:
            correct += 1
            status = "OK"
        else:
            status = "FAIL"
        
        print(f"  [{i}/{total}] {status} Expected: {expected}, Got: {predicted}")
    
    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    return accuracy

if __name__ == "__main__":
    questions = load_arc_easy()
    
    # Test 3B model
    print("="*50)
    print("3B MODEL ACCURACY")
    print("="*50)
    acc_3b = test_accuracy(MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf", questions)
    
    # Test 1.5B model
    print("\n" + "="*50)
    print("1.5B MODEL ACCURACY")
    print("="*50)
    acc_1_5b = test_accuracy(MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf", questions)
    
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"3B model accuracy:  {acc_3b:.2%}")
    print(f"1.5B model accuracy: {acc_1_5b:.2%}")
    print(f"Accuracy change: {(acc_1_5b - acc_3b)*100:+.1f}%")
