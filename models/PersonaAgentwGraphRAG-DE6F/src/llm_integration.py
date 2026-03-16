"""
LLM Integration module for PersonaAgent.
Demonstrates how to integrate the PersonaAgent with real LLM APIs like OpenAI GPT.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import json
import time
import requests
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from persona_agent import PersonaAgent, GraphRAGMemory, Interaction

# Optional imports for different LLM backends
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class RealPersonaAgent(PersonaAgent):
    """
    Enhanced PersonaAgent that integrates with real LLM APIs.
    Supports OpenAI GPT models and other compatible APIs.
    """
    
    def __init__(self, 
                 categories: List[str],
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo",
                 persona_weight: float = 0.4,
                 temperature: float = 0.3,
                 max_tokens: int = 50):
        """
        Initialize the Real LLM PersonaAgent.
        
        Args:
            categories: List of available categories
            api_key: OpenAI API key (optional, can use environment variable)
            model_name: LLM model to use
            persona_weight: Weight for persona influence
            temperature: LLM temperature for generation
            max_tokens: Maximum tokens for LLM response
        """
        super().__init__(categories, model_name, persona_weight)
        
        # Setup OpenAI client
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.has_api_key = bool(self.api_key)
        
        if not self.has_api_key:
            self.logger.warning("No OpenAI API key found. Using simulation mode.")
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the actual LLM API for inference.
        """
        if not self.has_api_key or not self.api_key:
            self.logger.info("API key missing!")
            return 
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies articles into categories. Always respond with just the category name, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=30
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Clean up the response to extract just the category
            prediction = prediction.lower()
            for category in self.categories:
                if category.lower() in prediction:
                    return category
            
            # If no exact match, try to find partial matches
            for category in self.categories:
                if any(word in prediction for word in category.lower().split()):
                    return category
            
            # Fallback to most common category if no match
            return "entertainment"
            
        except Exception as e:
            self.logger.warning(f"API call failed.")
            return
    
    def predict(self, user_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Make a personalized prediction using GraphRAG context and real LLM inference.
        """
        # Get semantic context from GraphRAG
        semantic_context = self.memory.get_semantic_context(user_id, query)
        
        # Generate personalized prompt
        personalized_prompt = self._generate_personalized_prompt(query, semantic_context)
        
        # Get LLM prediction using real API
        prediction = self._call_llm_api(personalized_prompt)
        
        # Ensure prediction is valid
        if prediction not in self.categories:
            # Fallback to most preferred category or default
            if semantic_context["category_preferences"]:
                prediction = max(semantic_context["category_preferences"].items(), 
                               key=lambda x: x[1])[0]
            else:
                prediction = "entertainment"
        
        metadata = {
            "semantic_context": semantic_context,
            "personalized_prompt": personalized_prompt,
            "graph_nodes": len(self.memory.nodes),
            "user_interactions": len(self.memory.user_interactions.get(user_id, [])),
            "api_used": self.has_api_key,
            "model": self.model_name
        }
        
        return prediction, metadata


class HuggingFacePersonaAgent(PersonaAgent):
    """
    PersonaAgent using HuggingFace Transformers for local LLM inference.
    Supports open-source models like Llama, Mistral, etc.
    """
    
    def __init__(self, 
                 categories: List[str],
                 model_name: str = "microsoft/DialoGPT-medium",
                 persona_weight: float = 0.4,
                 device: str = "auto",
                 max_new_tokens: int = 50):
        """
        Initialize HuggingFace PersonaAgent.
        
        Args:
            categories: List of available categories
            model_name: HuggingFace model name
            persona_weight: Weight for persona influence
            device: Device to run model on ('auto', 'cpu', 'cuda')
            max_new_tokens: Maximum tokens to generate
        """
        super().__init__(categories, model_name, persona_weight)
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("HuggingFace Transformers not available. Install with: pip install transformers torch")
            self.model_available = False
        else:
            self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        try:
            self.logger.info(f"Loading HuggingFace model: {self.model_name}")
            
            # Use text generation pipeline for simplicity
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            self.model_available = True
            self.logger.info("HuggingFace model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {e}")
            self.model_available = False
    
    def _call_huggingface_model(self, prompt: str) -> str:
        """Call HuggingFace model for inference."""
        if not self.model_available:
            self.logger.info("HuggingFace model not available")
            return 
        
        try:
            # Create a classification prompt
            classification_prompt = f"{prompt}\n\nCategory:"
            
            # Generate response
            outputs = self.pipeline(
                classification_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Extract category from response
            if "Category:" in generated_text:
                category_part = generated_text.split("Category:")[-1].strip().lower()
            else:
                category_part = generated_text.lower()
            
            # Find matching category
            for category in self.categories:
                if category.lower() in category_part:
                    return category
            
            # If no match, try partial matching
            for category in self.categories:
                if any(word in category_part for word in category.lower().split()):
                    return category
            
            return "entertainment"  # Default fallback
            
        except Exception as e:
            self.logger.warning(f"HuggingFace inference failed.")
            return 
    
    def predict(self, user_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using HuggingFace model."""
        semantic_context = self.memory.get_semantic_context(user_id, query)
        personalized_prompt = self._generate_personalized_prompt(query, semantic_context)
        
        prediction = self._call_huggingface_model(personalized_prompt)
        
        if prediction not in self.categories:
            if semantic_context["category_preferences"]:
                prediction = max(semantic_context["category_preferences"].items(), 
                               key=lambda x: x[1])[0]
            else:
                prediction = "entertainment"
        
        metadata = {
            "semantic_context": semantic_context,
            "personalized_prompt": personalized_prompt,
            "graph_nodes": len(self.memory.nodes),
            "user_interactions": len(self.memory.user_interactions.get(user_id, [])),
            "model_available": self.model_available,
            "model": self.model_name,
            "backend": "huggingface"
        }
        
        return prediction, metadata


class OllamaPersonaAgent(PersonaAgent):
    """
    PersonaAgent using Ollama for local LLM inference.
    Supports various open-source models through Ollama.
    """
    
    def __init__(self, 
                 categories: List[str],
                 model_name: str = "llama3",
                 persona_weight: float = 0.4,
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize Ollama PersonaAgent.
        
        Args:
            categories: List of available categories
            model_name: Ollama model name (e.g., 'llama2', 'mistral', 'codellama')
            persona_weight: Weight for persona influence
            ollama_host: Ollama server host
        """
        super().__init__(categories, model_name, persona_weight)
        
        self.ollama_host = ollama_host
        self.model_available = self._check_ollama_connection()
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name in model_names or any(self.model_name in name for name in model_names):
                    self.logger.info(f"Ollama server available with model {self.model_name}")
                    return True
                else:
                    self.logger.warning(f"Model {self.model_name} not found in Ollama. Available: {model_names}")
                    return False
            else:
                self.logger.warning("Ollama server not responding correctly")
                return False
                
        except Exception as e:
            self.logger.warning(f"Cannot connect to Ollama server: {e}")
            return False
    
    def _call_ollama_model(self, prompt: str) -> str:
        """Call Ollama model for inference."""
        if not self.model_available:
            self.logger.info("Ollama not available.")
            return 
        
        try:
            # Prepare the request
            data = {
                "model": self.model_name,
                "prompt": prompt + "\n\nAnswer with just the category name:",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 50
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=data,
                timeout=500
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip().lower()
                
                # Find matching category
                for category in self.categories:
                    if category.lower() in generated_text:
                        return category
                
                # Try partial matching
                for category in self.categories:
                    if any(word in generated_text for word in category.lower().split()):
                        return category
                
                return "entertainment"  # Default fallback
            else:
                self.logger.warning(f"Ollama request failed with status {response.status_code}")
                return 
                
        except Exception as e:
            self.logger.warning(f"Ollama inference failed: {e}.")
            return  
    
    def predict(self, user_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using Ollama model."""
        semantic_context = self.memory.get_semantic_context(user_id, query, task_type=getattr(self, 'task_type', 'news'))
        personalized_prompt = self._generate_personalized_prompt(query, semantic_context)
        
        prediction = self._call_ollama_model(personalized_prompt)
        
        if prediction not in self.categories:
            if semantic_context.get("category_preferences"):
                prediction = max(semantic_context["category_preferences"].items(), 
                               key=lambda x: x[1])[0]
            else:
                prediction = "entertainment"
        
        metadata = {
            "semantic_context": semantic_context,
            "personalized_prompt": personalized_prompt,
            "graph_nodes": len(self.memory.nodes),
            "user_interactions": len(self.memory.user_interactions.get(user_id, [])),
            "model_available": self.model_available,
            "model": self.model_name,
            "backend": "ollama",
            "host": self.ollama_host
        }
        
        return prediction, metadata


class GenericAPIPersonaAgent(PersonaAgent):
    """
    PersonaAgent using a generic REST API for LLM inference.
    Compatible with OpenAI-style APIs, LocalAI, and other compatible services.
    """
    
    def __init__(self, 
                 categories: List[str],
                 api_endpoint: str,
                 model_name: str = "gpt-3.5-turbo",
                 persona_weight: float = 0.4,
                 api_key: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 50):
        """
        Initialize Generic API PersonaAgent.
        
        Args:
            categories: List of available categories
            api_endpoint: API endpoint URL
            model_name: Model name to use
            persona_weight: Weight for persona influence
            api_key: Optional API key
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(categories, model_name, persona_weight)
        
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_available = self._check_api_connection()
    
    def _check_api_connection(self) -> bool:
        """Check if the API endpoint is available."""
        try:
            # Try a simple request to check connectivity
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(self.api_endpoint.rstrip('/') + '/models', 
                                  headers=headers, timeout=5)
            
            if response.status_code in [200, 404]:  # 404 is OK, means server is up
                self.logger.info(f"Generic API endpoint available: {self.api_endpoint}")
                return True
            else:
                self.logger.warning(f"API endpoint returned status {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Cannot connect to API endpoint: {e}")
            return False
    
    def _call_generic_api(self, prompt: str) -> str:
        """Call generic API for inference."""
        if not self.model_available:
            self.logger.info("Generic API not available.")
            return  
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that classifies articles into categories. Always respond with just the category name, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                self.api_endpoint.rstrip('/') + '/chat/completions',
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['choices'][0]['message']['content'].strip().lower()
                
                # Find matching category
                for category in self.categories:
                    if category.lower() in prediction:
                        return category
                
                # Try partial matching
                for category in self.categories:
                    if any(word in prediction for word in category.lower().split()):
                        return category
                
                return "entertainment"
            else:
                self.logger.warning(f"API request failed with status {response.status_code}")
                return  
                
        except Exception as e:
            self.logger.warning(f"Generic API inference failed: {e}.")
            return 
    
    def predict(self, user_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using generic API."""
        semantic_context = self.memory.get_semantic_context(user_id, query)
        personalized_prompt = self._generate_personalized_prompt(query, semantic_context)
        
        prediction = self._call_generic_api(personalized_prompt)
        
        if prediction not in self.categories:
            if semantic_context["category_preferences"]:
                prediction = max(semantic_context["category_preferences"].items(), 
                               key=lambda x: x[1])[0]
            else:
                prediction = "entertainment"
        
        metadata = {
            "semantic_context": semantic_context,
            "personalized_prompt": personalized_prompt,
            "graph_nodes": len(self.memory.nodes),
            "user_interactions": len(self.memory.user_interactions.get(user_id, [])),
            "model_available": self.model_available,
            "model": self.model_name,
            "backend": "generic_api",
            "endpoint": self.api_endpoint
        }
        
        return prediction, metadata


class AdaptivePersonaAgent(OllamaPersonaAgent):
    """
    Advanced PersonaAgent with adaptive learning capabilities.
    Continuously improves based on user feedback and corrections.
    """
    
    def __init__(self, 
                 categories: List[str],
                 model_name: str = "llama3",
                 ollama_host: str = "http://localhost:11434",
                 persona_weight: float = 0.4,
                 task_type: str = "news",
                 task_config: Dict[str, str] = None):
        """
        Initialize the Adaptive PersonaAgent.
        
        Args:
            categories: List of available categories
            model_name: Model name for Ollama
            ollama_host: Ollama server host
            persona_weight: Weight for persona influence (0-1)
            task_type: Type of task ('news', 'movie', 'custom')
            task_config: Custom task configuration for prompts
        """
        super().__init__(categories, model_name, persona_weight, ollama_host)
        
        # Override the base agent with task-specific configuration
        self.task_type = task_type
        self.task_config = task_config or self._get_default_task_config(task_type)
        
        self.feedback_history: Dict[str, List[Dict[str, Any]]] = {}
        self.adaptation_rate = 0.1
    
    def _get_default_task_config(self, task_type: str) -> Dict[str, str]:
        """Get default task configuration based on task type."""
        configs = {
            "news": {
                "content_type": "article",
                "classification_term": "categories", 
                "content_key": "article:",
                "category_term": "Category",
                "preference_term": "category preferences",
                "interaction_type": "articles",
                "instruction": "Based on the article content and the metadata, classify this article.\nConsider user's personal interactions, similar interactions from other users, and relevant concepts as references. Give higher weight to interactions with higher similarity scores.\nAnswer with just the category name."
            },
            "movie": {
                "content_type": "movie",
                "classification_term": "tags",
                "content_key": "description:",
                "category_term": "Tag", 
                "preference_term": "tag preferences",
                "interaction_type": "movies",
                "instruction": "Based on the movie description and the metadata, classify this movie.\nConsider user's personal interactions, similar interactions from other users, and relevant concepts as references. Give higher weight to interactions with higher similarity scores.\nAnswer with just the tag name."
            },
            "product_rating": {
                "content_type": "product review",
                "classification_term": "ratings",
                "content_key": "review:",
                "category_term": "Rating",
                "preference_term": "rating preferences",
                "interaction_type": "product reviews",
                "instruction": "Based on the product review text and the metadata, predict the rating (1-5 stars).\nConsider user's personal review patterns, similar reviews from other users, and relevant concepts as references. Give higher weight to reviews with higher similarity scores.\nAnswer with just the rating number (1, 2, 3, 4, or 5)."
            },
            "custom": {
                "content_type": "item",
                "classification_term": "categories",
                "content_key": "content:",
                "category_term": "Category",
                "preference_term": "category preferences", 
                "interaction_type": "items",
                "instruction": "Based on the content and the metadata, classify this item.\nConsider user's personal interactions, similar interactions from other users, and relevant concepts as references. Give higher weight to interactions with higher similarity scores.\n\nAnswer with just the category name."
            }
        }
        return configs.get(task_type, configs["custom"])
    
    def _generate_personalized_prompt(self, 
                                    query: str, 
                                    semantic_context: Dict[str, Any]) -> str:
        """Generate a personalized prompt incorporating both user and global GraphRAG context."""
        
        # Extract content based on task configuration
        content_key = self.task_config["content_key"]
        if content_key in query:
            content_text = query.split(content_key)[-1].strip()
        else:
            content_text = query
        
        # Build prompt using task configuration
        base_prompt = f"""
Task: Classify the following {self.task_config['content_type']} into one of these {self.task_config['classification_term']}: {', '.join(self.categories)}

{self.task_config['content_type'].title()}: {content_text}

PERSONALIZATION CONTEXT:
"""
        
        # Add user's own relevant interactions (sorted by similarity score)
        if semantic_context.get("user_interactions"):
            user_interactions = semantic_context["user_interactions"][:3]  # Show top 3
            base_prompt += f"\nSimilar {self.task_config['interaction_type']} the user has personally interacted with (ordered by relevance):\n"
            for i, interaction in enumerate(user_interactions, 1):
                score = interaction['relevance_score']
                confidence = "HIGH" if score > 0.5 else "MEDIUM" if score > 0.2 else "LOW"
                base_prompt += f"{i}. \"{interaction['title']}\" ({self.task_config['category_term']}: {interaction['category']}) - Similarity: {score:.3f} ({confidence})\n"
        
        # Add global relevant interactions from other users (sorted by similarity score)
        if semantic_context.get("global_interactions"):
            global_interactions = semantic_context["global_interactions"][:3]  # Show top 3
            base_prompt += f"\nSimilar {self.task_config['interaction_type']} from other users with similar interests (ordered by relevance):\n"
            for i, interaction in enumerate(global_interactions, 1):
                score = interaction['relevance_score']
                confidence = "HIGH" if score > 0.5 else "MEDIUM" if score > 0.2 else "LOW"
                base_prompt += f"{i}. \"{interaction['title']}\" ({self.task_config['category_term']}: {interaction['category']}) - Similarity: {score:.3f} ({confidence})\n"
        
        # Add category preferences
        if semantic_context.get("category_preferences"):
            top_prefs = sorted(semantic_context["category_preferences"].items(), 
                             key=lambda x: x[1], reverse=True)[:3]
            prefs_str = ", ".join([f"{cat} ({pref:.2f})" for cat, pref in top_prefs])
            base_prompt += f"\nUser's {self.task_config['preference_term']}: {prefs_str}\n"
        
        # Add relevant concepts
        if semantic_context.get("relevant_concepts"):
            concepts_str = ", ".join(semantic_context["relevant_concepts"][:10])
            base_prompt += f"\nRelevant concepts from interaction history: {concepts_str}\n"
        
        # Add interaction statistics
        if semantic_context.get("interaction_count"):
            base_prompt += f"\nUser interaction history: {semantic_context['interaction_count']} total interactions\n"
        
        # Add enhanced task-specific instruction with similarity score emphasis
        base_prompt += f"\nCLASSIFICATION GUIDANCE:\n"
        base_prompt += f"- Pay special attention to interactions with HIGH similarity scores (>0.5) as they are most relevant\n"
        base_prompt += f"- Consider MEDIUM similarity scores (0.2-0.5) as moderately relevant\n"
        base_prompt += f"- Use LOW similarity scores (<0.2) only as supporting context\n"
        base_prompt += f"- Prioritize user's personal interactions over global interactions when similarity scores are comparable\n"
        
        instruction = self.task_config['instruction']
        
        base_prompt += f"\n{instruction}"
        
        return base_prompt
    
    def add_feedback(self, user_id: str, query: str, predicted: str, actual: str, helpful: bool = True):
        """
        Add user feedback to improve future predictions.
        
        Args:
            user_id: User identifier
            query: Original query
            predicted: What the system predicted
            actual: What the actual correct answer was
            helpful: Whether the personalization was helpful
        """
        if user_id not in self.feedback_history:
            self.feedback_history[user_id] = []
        
        feedback = {
            "query": query,
            "predicted": predicted,
            "actual": actual,
            "helpful": helpful,
            "timestamp": time.time()
        }
        
        self.feedback_history[user_id].append(feedback)
        
        # Update user profile with corrected interaction
        if predicted != actual:
            corrected_interaction = Interaction(
                id=f"feedback_{len(self.feedback_history[user_id])}",
                text=query.split('article: ')[-1] if 'article: ' in query else query,
                title="User Corrected Classification",
                category=actual
            )
            self.memory.add_interaction(user_id, corrected_interaction)
    
    def get_user_adaptation_stats(self, user_id: str) -> Dict[str, Any]:
        """Get adaptation statistics for a user."""
        if user_id not in self.feedback_history:
            return {"total_feedback": 0, "accuracy_trend": [], "helpful_rate": 0.0}
        
        feedback = self.feedback_history[user_id]
        total_feedback = len(feedback)
        
        if total_feedback == 0:
            return {"total_feedback": 0, "accuracy_trend": [], "helpful_rate": 0.0}
        
        # Calculate accuracy trend over time
        correct_predictions = [1 if f["predicted"] == f["actual"] else 0 for f in feedback]
        helpful_count = sum(1 for f in feedback if f["helpful"])
        
        return {
            "total_feedback": total_feedback,
            "accuracy_trend": correct_predictions,
            "helpful_rate": helpful_count / total_feedback,
            "recent_accuracy": sum(correct_predictions[-10:]) / min(10, len(correct_predictions))
        }


def demo_open_source_llm_backends():
    """Demo all available open-source LLM backends."""
    print("🔗 Open Source LLM Backends Demo")
    print("=" * 60)
    
    categories = [
    "women",
    "religion",
    "politics",
    "style & beauty",
    "entertainment",
    "culture & arts",
    "sports",
    "science & technology",
    "travel",
    "business",
    "crime",
    "education",
    "healthy living",
    "parents",
    "food & drink"
]
    
    # Sample user interactions
    sample_interactions = [
        {"text": "Concert review and music analysis", "title": "Music News", "category": "entertainment"},
        {"text": "Basketball championship finals", "title": "Sports Update", "category": "sports"},
        {"text": "New smartphone features unveiled", "title": "Tech Review", "category": "science & technology"}
    ]
    
    user_id = "demo_user"
    
    # Initialize different backend agents
    agents = {}
    
    print("\n🤖 Initializing different LLM backends...")
    
    # 1. Standard simulation agent
    agents["Simulation"] = PersonaAgent(categories=categories)
    print("   ✓ Simulation agent ready")
    
    # 2. OpenAI agent (if available)
    if OPENAI_AVAILABLE:
        agents["OpenAI"] = RealPersonaAgent(
            categories=categories,
            model_name="gpt-3.5-turbo"
        )
        print(f"   {'✓' if agents['OpenAI'].has_api_key else '⚠'} OpenAI agent {'ready' if agents['OpenAI'].has_api_key else 'using simulation (no API key)'}")
    else:
        print("   ⚠ OpenAI not available (pip install openai)")
    
    # 3. HuggingFace agent (if available)
    if TRANSFORMERS_AVAILABLE:
        agents["HuggingFace"] = HuggingFacePersonaAgent(
            categories=categories,
            model_name="microsoft/DialoGPT-small",  # Smaller model for demo
            device="cpu"
        )
        print(f"   {'✓' if agents['HuggingFace'].model_available else '⚠'} HuggingFace agent {'ready' if agents['HuggingFace'].model_available else 'failed to load model'}")
    else:
        print("   ⚠ HuggingFace not available (pip install transformers torch)")
    
    # 4. Ollama agent (if server is running)
    agents["Ollama"] = OllamaPersonaAgent(
        categories=categories,
        model_name="llama2"
    )
    print(f"   {'✓' if agents['Ollama'].model_available else '⚠'} Ollama agent {'ready' if agents['Ollama'].model_available else 'server not available'}")
    
    # 5. Generic API agent (LocalAI example)
    agents["LocalAI"] = GenericAPIPersonaAgent(
        categories=categories,
        api_endpoint="http://localhost:8080",
        model_name="gpt-3.5-turbo"
    )
    print(f"   {'✓' if agents['LocalAI'].model_available else '⚠'} LocalAI agent {'ready' if agents['LocalAI'].model_available else 'server not available'}")
    
    # Add user interactions to all agents
    print("\n🧠 Building GraphRAG memory for all agents...")
    for name, agent in agents.items():
        agent.add_user_interactions(user_id, sample_interactions)
    
    # Test queries
    test_queries = [
        "Article: Pop star announces world tour dates and new album release.",
        "Article: Machine learning breakthrough enables faster drug discovery.",
        "Article: Stock market volatility affects global trading patterns."
    ]
    
    expected_categories = ["entertainment", "science & technology", "business"]
    
    print("\n📊 Backend Comparison Results:")
    print("=" * 80)
    header = f"{'Query':<35}"
    for name in agents.keys():
        header += f"{name:<12}"
    print(header)
    print("-" * 80)
    
    results = {}
    for i, (query, expected) in enumerate(zip(test_queries, expected_categories)):
        query_short = query.split("Article: ")[1][:30] + "..."
        row = f"{query_short:<35}"
        results[i] = {"query": query, "expected": expected, "predictions": {}}
        
        for name, agent in agents.items():
            try:
                prediction, metadata = agent.predict(user_id, query)
                results[i]["predictions"][name] = {
                    "prediction": prediction,
                    "metadata": metadata
                }
                status = "✓" if prediction == expected else "✗"
                row += f"{prediction[:8]:<3} {status:<1} "
            except Exception as e:
                row += f"{'ERROR':<8} ✗ "
                results[i]["predictions"][name] = {"prediction": "ERROR", "error": str(e)}
        
        print(row)
    
    # Show detailed analysis
    print("\n📈 Detailed Analysis:")
    print("=" * 60)
    
    for name, agent in agents.items():
        if name not in ["Simulation"]:  # Skip basic simulation for detailed analysis
            print(f"\n🔍 {name} Backend Analysis:")
            
            # Show availability status
            if hasattr(agent, 'model_available'):
                print(f"   Model Available: {'✓' if agent.model_available else '✗'}")
            if hasattr(agent, 'has_api_key'):
                print(f"   API Key Available: {'✓' if agent.has_api_key else '✗'}")
            
            # Show backend-specific info
            if hasattr(agent, 'backend'):
                print(f"   Backend Type: {getattr(agent, 'backend', 'unknown')}")
            if hasattr(agent, 'ollama_host'):
                print(f"   Ollama Host: {agent.ollama_host}")
            if hasattr(agent, 'api_endpoint'):
                print(f"   API Endpoint: {agent.api_endpoint}")
            
            # Show a sample prediction metadata
            if results and name in results[0]["predictions"]:
                metadata = results[0]["predictions"][name].get("metadata", {})
                print(f"   Graph Nodes: {metadata.get('graph_nodes', 'N/A')}")
                print(f"   User Interactions: {metadata.get('user_interactions', 'N/A')}")
    
    print(f"\n💡 Setup Instructions:")
    print("=" * 60)
    print("🔑 OpenAI: Set OPENAI_API_KEY environment variable")
    print("🤗 HuggingFace: pip install transformers torch")
    print("🦙 Ollama: Install Ollama (https://ollama.ai/) and run 'ollama pull llama2'")
    print("🔧 LocalAI: Run LocalAI server on localhost:8080")
    print("🌐 Generic API: Use with any OpenAI-compatible API endpoint")
    
    return results


def demo_real_llm_integration():
    """Demo the real LLM integration capabilities."""
    print("🔗 Real LLM Integration Demo")
    print("=" * 50)
    
    # Categories for news classification
    categories = [
        "women", "religion", "politics", "style & beauty", "entertainment", 
        "culture & arts", "sports", "science & technology", "travel", 
        "business", "crime", "education", "healthy living", "parents", "food & drink"
    ]
    
    # Initialize agents
    print("\n🤖 Initializing agents...")
    
    # Standard agent with simulation
    standard_agent = PersonaAgent(categories=categories)
    
    # Real LLM agent (will use simulation if no API key)
    real_llm_agent = OllamaPersonaAgent(
    categories=categories,
    model_name="llama2",  # or "mistral", "codellama"
    ollama_host="http://localhost:11434"
)
    
    # Adaptive agent
    adaptive_agent = AdaptivePersonaAgent(
        categories=categories,
        model_name="llama2",  # or "mistral", "codellama"
        ollama_host="http://localhost:11434"
    )
    
    # Sample user data
    sample_interactions = [
        {"text": "Harry Styles announces new album", "title": "Music News", "category": "entertainment"},
        {"text": "Olympic games highlights", "title": "Sports Update", "category": "sports"},
        {"text": "Latest iPhone review", "title": "Tech Review", "category": "science & technology"},
        {"text": "Fashion week trends", "title": "Style News", "category": "style & beauty"}
    ]
    
    user_id = "demo_user"
    
    # Add interactions to all agents
    for agent in [standard_agent, real_llm_agent, adaptive_agent]:
        agent.add_user_interactions(user_id, sample_interactions)
    
    # Test queries
    test_queries = [
        "Which category does this article relate to? Article: Taylor Swift releases new song featuring pop elements.",
        "Which category does this article relate to? Article: Scientists discover new treatment for diabetes.",
        "Which category does this article relate to? Article: Stock market reaches all-time high today."
    ]
    
    print("\n📊 Comparison Results:")
    print("-" * 70)
    print(f"{'Query':<40} {'Standard':<15} {'Real LLM':<15} {'Adaptive':<15}")
    print("-" * 70)
    
    for i, query in enumerate(test_queries):
        # Get predictions from all agents
        pred_standard, _ = standard_agent.predict(user_id, query)
        pred_real, meta_real = real_llm_agent.predict(user_id, query)
        pred_adaptive, _ = adaptive_agent.predict(user_id, query)
        
        query_short = query.split("Article: ")[1][:35] + "..." if "Article: " in query else query[:35] + "..."
        
        print(f"{query_short:<40} {pred_standard:<15} {pred_real:<15} {pred_adaptive:<15}")
        
        # Simulate user feedback for adaptive agent
        if i == 0:  # Assume first prediction was wrong, correct it
            adaptive_agent.add_feedback(user_id, query, pred_adaptive, "entertainment", helpful=True)
    
    # Show adaptive agent stats
    print(f"\n🧠 Adaptive Agent Learning Stats:")
    stats = adaptive_agent.get_user_adaptation_stats(user_id)
    print(f"   Total feedback: {stats['total_feedback']}")
    print(f"   Helpful rate: {stats['helpful_rate']:.2f}")
    
    print("\n✅ Real LLM Integration demo complete!")


def example_production_usage():
    """Show example of how to use PersonaAgent in production."""
    print("\n🏭 Production Usage Example")
    print("=" * 50)
    
    # Load your categories (from config file in real scenario)
    categories = ["entertainment", "sports", "politics", "technology", "business"]
    
    # Initialize production agent
    agent = AdaptivePersonaAgent(
        categories=categories,
        model_name="llama2",  # or "mistral", "codellama"
        ollama_host="http://localhost:11434"
    )
    
    # Example: New user registration
    user_id = "user_12345"
    
    # Example: User interacts with content
    user_interactions = [
        {"text": "Breaking: Elections results announced", "title": "Political News", "category": "politics"},
        {"text": "New startup raises $100M", "title": "Business Update", "category": "business"}
    ]
    
    agent.add_user_interactions(user_id, user_interactions)
    
    # Example: Make prediction for new content
    new_content = "Which category does this relate to? Article: Tech company announces AI breakthrough."
    
    prediction, metadata = agent.predict(user_id, new_content)
    
    print(f"📝 Production Example:")
    print(f"   User: {user_id}")
    print(f"   Content: {new_content.split('Article: ')[1]}")
    print(f"   Prediction: {prediction}")
    print(f"   Confidence indicators:")
    print(f"     - User interactions: {metadata['user_interactions']}")
    print(f"     - Graph nodes: {metadata['graph_nodes']}")
    
    # Example: User provides feedback
    actual_category = "technology"  # What user confirms
    agent.add_feedback(user_id, new_content, prediction, actual_category, helpful=True)
    
    print(f"   User feedback: Actual category was '{actual_category}'")
    print(f"   System will learn from this feedback for future predictions")


if __name__ == "__main__":
    print("🚀 LLM Integration Module for PersonaAgent")
    
    # Run demos
    demo_open_source_llm_backends()
    demo_real_llm_integration()
    example_production_usage()
    
    print("\n" + "=" * 60)
    print("✨ Integration demos complete!")
    print("   Ready for production deployment with open-source and commercial LLM APIs")
