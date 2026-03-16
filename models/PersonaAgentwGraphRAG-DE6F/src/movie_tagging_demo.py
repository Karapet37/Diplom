import sys
import os

from llm_integration import *
from persona_agent import PersonaAgent
from movie_tagging_dataset_config import get_dataset_config
from configurable_demo import *


# Define movie categories
categories = [
    "sci-fi", "based on a book", "comedy", "action", "twist ending", 
    "dystopia", "dark comedy", "classic", "psychology", "fantasy", 
    "romance", "thought-provoking", "social commentary", "violence", "true story"
]

# Initialize agent with local Llama3 model
print("\n🔧 Initializing PersonaAgent with local Llama3...")
agent = AdaptiveHuggingFacePersonaAgent(
    categories=categories,
    model_name="./models/llama3-8b-instruct",  # Your local model
    device="auto",  # or "cuda" if you have GPU
    task_type="movie",
    max_new_tokens=50
)


# Load movie data (only first 5 users for demo)
print("📁 Loading movie data...")
movie_config = get_dataset_config('movies')
train_data, test_data, labels = movie_config.load_data()


print(f"   ✓ Using {len(train_data)} training users for demo (limited to 5)")
print(f"   ✓ Using {len(test_data)} test users for demo (limited to 5)")
print(f"   ✓ Total labels available: {len(labels)}")

# Add user interactions to build GraphRAG memory
print("\n🧠 Building GraphRAG memory from movie viewing histories...")
total_interactions = 0
for i, example in enumerate(train_data):
    user_id = str(example.get('user_id', 'unknown'))
    profile = example.get('profile', [])
    
    print(f"   User {user_id}: {len(profile)} movies in profile")
    
    # Convert profile to standard format
    standard_profile = movie_config.convert_profile_to_standard(profile)
    agent.add_user_interactions(user_id, standard_profile)
    total_interactions += len(standard_profile)

for i, example in enumerate(test_data):
    user_id = str(example.get('user_id', 'unknown'))
    profile = example.get('profile', [])
    
    print(f"   User {user_id}: {len(profile)} movies in profile")
    
    # Convert profile to standard format
    standard_profile = movie_config.convert_profile_to_standard(profile)
    agent.add_user_interactions(user_id, standard_profile)
    
    # Show first movie for verification
    if profile:
        first_movie = profile[0]
        print(f"     First movie: {first_movie.get('tag', 'unknown')} - {first_movie.get('description', '')[:50]}...")

print(f"\n   ✓ GraphRAG memory built with {len(agent.memory.user_interactions)} users")
print(f"   ✓ Total interactions added: {total_interactions}")
print(f"   ✓ Total graph nodes: {len(agent.memory.nodes)}")


prompts = []
labels = []
results = []
pipelines = pipeline(
                    "text-generation",
                    model="./models/llama3-8b-instruct",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
for i, example in enumerate(test_data):
    user_id = str(example.get('user_id', 'unknown'))
    profile = example.get('profile', [])
    
    print(f"   User {user_id}: {len(profile)} movies in profile")
    
    # Convert profile to standard format
    standard_profile = movie_config.convert_profile_to_standard(profile)
    
    # Process each movie in the user's profile
    for j, items in enumerate(standard_profile):
            
        label = items['category']
        query = items['text']
        
        print(f"\n   --- Movie {j+1} for User {user_id} ---")
        print(f"   True Label: {label}")
        print(f"   Query: {query[:100]}...")
        
        # Get semantic context for this specific query
        semantic_context = agent.memory.get_semantic_context(user_id, query)
        
        # Build the complete prompt
        base_prompt = f"""Task: Classify the following movie into one of these tags: sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story.

Movie description: {query}

Also here are additioanl CONTEXT:
"""
        
        # Add user's own relevant interactions (sorted by similarity score)
        if semantic_context.get("user_interactions"):
            user_interactions = semantic_context["user_interactions"][:3]  # Show top 3
            base_prompt += f"\nSimilar movies the user has personally interacted with (ordered by relevance):\n"
            for k, interaction in enumerate(user_interactions, 1):
                score = interaction['relevance_score']
                confidence = "HIGH" if score > 0.5 else "MEDIUM" if score > 0.2 else "LOW"
                base_prompt += f"{k}. \"{interaction['text']}\" (Tag: {interaction['category']}) - Similarity: {score:.3f} ({confidence})\n"
        else:
            base_prompt += f"\nSimilar movies the user has personally interacted with (ordered by relevance):\n"
            base_prompt += f"1. (No similar movies found in user's history)\n"
        
        # Add global relevant interactions from other users (sorted by similarity score)
        if semantic_context.get("global_interactions"):
            global_interactions = semantic_context["global_interactions"][:3]  # Show top 3
            base_prompt += f"\nSimilar movies from other users with similar interests (ordered by relevance):\n"
            for k, interaction in enumerate(global_interactions, 1):
                score = interaction['relevance_score']
                confidence = "HIGH" if score > 0.5 else "MEDIUM" if score > 0.2 else "LOW"
                # Handle empty titles for movie data
                title_display = interaction['title'] if interaction['title'].strip() else interaction['text']
                base_prompt += f"{k}. \"{title_display}\" (Tag: {interaction['category']}) - Similarity: {score:.3f} ({confidence})\n"
        else:
            base_prompt += f"\nSimilar movies from other users with similar interests (ordered by relevance):\n"
            base_prompt += f"1. (No similar movies found from other users)\n"
        
        # Add category preferences
        if semantic_context.get("category_preferences"):
            top_prefs = sorted(semantic_context["category_preferences"].items(), 
                             key=lambda x: x[1], reverse=True)[:3]
            prefs_str = ", ".join([f"{cat} ({pref:.2f})" for cat, pref in top_prefs])
            base_prompt += f"\nUser's tag preferences: {prefs_str}\n"
        else:
            base_prompt += f"\nUser's tag preferences: (No preferences established yet)\n"
        
        # Add relevant concepts
        if semantic_context.get("relevant_concepts"):
            concepts_str = ", ".join(semantic_context["relevant_concepts"][:10])
            base_prompt += f"\nRelevant concepts from interaction history: {concepts_str}\n"
        
        # Add interaction statistics
        if semantic_context.get("interaction_count"):
            base_prompt += f"\nUser interaction history: {semantic_context['interaction_count']} total interactions\n"
        
        # Add classification guidance and instruction
        base_prompt += f"""
CLASSIFICATION GUIDANCE:
- Pay special attention to interactions with HIGH similarity scores (>0.5) as they are most relevant
- Consider MEDIUM similarity scores (0.2-0.5) as moderately relevant
- Use LOW similarity scores (<0.2) only as supporting context

Based on the movie description and tags from similar movies, classify this movie.

Answer with just the tag name."""
        
        print(f"\n   GENERATED PROMPT:")
        print("   " + "="*80)
        print(base_prompt)
        print("   " + "="*80)
        prompts.append(base_prompt)
        labels.append(label)
        
        try:
            
            # Generate response
            
            outputs = pipelines(
                base_prompt,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=pipelines.tokenizer.eos_token_id
            )
            
            # Extract generated text
            llama_prediction = outputs[0]['generated_text']
            
            results.append({
                'user_id': user_id,
                'movie_index': j,
                'true_label': label,
                'llama_prediction': llama_prediction,
                # 'correct': llama_prediction == label,
                'prompt': base_prompt
            })
            
        except Exception as e:
            print(f"❌ Error calling Llama: {e}")
            results.append({
                'user_id': user_id,
                'movie_index': j,
                'true_label': label,
                'llama_prediction': 'ERROR',
                # 'correct': False,
                'prompt': base_prompt
            })

## save results and calculate accuracy
import json
from datetime import datetime

# Calculate accuracy
# total_predictions = len(results)
# correct_predictions = sum(1 for r in results if r['correct'])
# accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

# print(f"\n📊 Results Summary:")
# print(f"Total predictions: {total_predictions}")
# print(f"Correct predictions: {correct_predictions}")
# print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

# Save results to JSON
results_data = {
    # 'timestamp': datetime.now().isoformat(),
    # 'model': './models/llama3-8b-instruct',
    # 'total_predictions': total_predictions,
    # 'correct_predictions': correct_predictions,
    # 'accuracy': accuracy,
    'results': results
}

with open('llama_results2.json', 'w', encoding='utf-8') as f:
    json.dump(results_data, f, indent=2, ensure_ascii=False)

print(f"💾 Results saved to llama_results.json")

# Save CSV summary
# import csv
# with open('results_summary2.csv', 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['User_ID', 'Movie_Index', 'True_Label', 'Prediction', 'Correct'])
    
#     for result in results:
#         writer.writerow([
#             result['user_id'],
#             result['movie_index'],
#             result['true_label'],
#             result['llama_prediction'],
#             'Yes' if result['correct'] else 'No'
#         ])

# print(f"💾 CSV saved to results_summary.csv")
# print(f"✅ All results saved!")
