"""
PersonaAgent: LLM-based personalized citation identification system with GraphRAG memory.
Leverages users' past interactions stored in a graph-based retrieval system for improved inference.
"""

import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from datetime import datetime
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


@dataclass
class Interaction:
    """Represents a single user interaction."""
    id: str
    text: str
    title: str
    category: str
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """Represents a node in the GraphRAG system."""
    node_id: str
    node_type: str  # 'interaction', 'concept', 'category', 'entity'
    content: str
    properties: Dict[str, Any] = field(default_factory=dict)
    connections: Set[str] = field(default_factory=set)


class GraphRAGMemory:
    """
    Graph-based Retrieval-Augmented Generation memory system for storing and retrieving
    user interactions with semantic relationships.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize the GraphRAG memory system."""
        self.graph = nx.Graph()
        self.nodes: Dict[str, GraphNode] = {}
        self.user_interactions: Dict[str, List[str]] = defaultdict(list)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.interaction_texts = []
        self.interaction_ids = []
        self.embedding_dim = embedding_dim
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities and key concepts from text."""
        # Simple entity extraction - can be enhanced with NER models
        text_lower = text.lower()
        entities = []
        
        # Extract capitalized words (potential named entities)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(words)
        
        # Extract common concepts
        concept_patterns = [
            r'\b(music|movie|film|album|song|artist|actor|actress)\b',
            r'\b(sports|game|team|player|championship|olympics)\b',
            r'\b(politics|election|government|president|policy)\b',
            r'\b(technology|tech|AI|software|app|digital)\b',
            r'\b(travel|destination|vacation|trip|country|city)\b',
            r'\b(fashion|style|beauty|clothing|makeup)\b',
            r'\b(food|recipe|restaurant|cuisine|cooking)\b',
            r'\b(health|medical|fitness|wellness|diet)\b',
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text_lower)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _create_interaction_node(self, interaction: Interaction, user_id: str) -> GraphNode:
        """Create a graph node for an interaction."""
        # Create unique node ID that includes user_id to prevent collisions
        unique_node_id = f"interaction_{user_id}_{interaction.id}"
        node = GraphNode(
            node_id=unique_node_id,
            node_type="interaction",
            content=f"{interaction.title} {interaction.text}",
            properties={
                "category": interaction.category,
                "title": interaction.title,
                "text": interaction.text,
                "timestamp": interaction.timestamp.isoformat(),
                "interaction_id": interaction.id,
                "user_id": user_id  # Store user_id for debugging
            }
        )
        return node
    
    def _create_concept_node(self, concept: str) -> GraphNode:
        """Create a graph node for a concept or entity."""
        node_id = f"concept_{hashlib.md5(concept.encode()).hexdigest()[:8]}"
        node = GraphNode(
            node_id=node_id,
            node_type="concept",
            content=concept,
            properties={"name": concept}
        )
        return node
    
    def _create_category_node(self, category: str) -> GraphNode:
        """Create a graph node for a category."""
        node_id = f"category_{category.replace(' ', '_').replace('&', 'and')}"
        node = GraphNode(
            node_id=node_id,
            node_type="category",
            content=category,
            properties={"name": category}
        )
        return node
    
    def add_interaction(self, user_id: str, interaction: Interaction):
        """Add a user interaction to the GraphRAG memory."""
        # Create interaction node with user-specific ID to avoid collisions
        interaction_node = self._create_interaction_node(interaction, user_id)
        self.nodes[interaction_node.node_id] = interaction_node
        self.graph.add_node(interaction_node.node_id, **interaction_node.properties)
        
        # Track user interactions
        self.user_interactions[user_id].append(interaction_node.node_id)
        
        # Create category node if not exists
        category_node_id = f"category_{interaction.category.replace(' ', '_').replace('&', 'and')}"
        if category_node_id not in self.nodes:
            category_node = self._create_category_node(interaction.category)
            self.nodes[category_node_id] = category_node
            self.graph.add_node(category_node_id, **category_node.properties)
        
        # Connect interaction to category
        self.graph.add_edge(interaction_node.node_id, category_node_id, relation="belongs_to")
        
        # Extract and connect concepts
        entities = self._extract_entities(interaction.text + " " + interaction.title)
        for entity in entities:
            concept_node_id = f"concept_{hashlib.md5(entity.encode()).hexdigest()[:8]}"
            
            if concept_node_id not in self.nodes:
                concept_node = self._create_concept_node(entity)
                self.nodes[concept_node_id] = concept_node
                self.graph.add_node(concept_node_id, **concept_node.properties)
            
            # Connect interaction to concept
            self.graph.add_edge(interaction_node.node_id, concept_node_id, relation="contains")
        
        # Update text corpus for TF-IDF
        self.interaction_texts.append(interaction.text + " " + interaction.title)
        self.interaction_ids.append(interaction_node.node_id)
        
        # Refit TF-IDF if we have enough samples
        if len(self.interaction_texts) > 1:
            try:
                self.tfidf_vectorizer.fit(self.interaction_texts)
            except ValueError:
                pass  # Handle case where all texts are too similar
    
    def retrieve_relevant_interactions(self, 
                                     user_id: str, 
                                     query: str,
                                     task_type: str = "news", 
                                     top_k: int = 5) -> Dict[str, List[Tuple[Interaction, float]]]:
        """
        Enhanced retrieval of relevant interactions based on extracted content.
        
        Args:
            user_id: User identifier
            query: Query text to find relevant interactions for
            task_type: Type of task ('news', 'movie') to determine content extraction
            top_k: Number of top interactions to retrieve
            
        Returns:
            Dictionary with 'user_interactions' and 'global_interactions' keys
        """
        if not self.interaction_texts:
            return {"user_interactions": [], "global_interactions": []}
        
        try:
            # Step 1: Extract relevant content from query based on task type
            extracted_content = self._extract_content_from_query(query, task_type)
            if not extracted_content:
                extracted_content = query  # Fallback to full query
            
            # Get TF-IDF representation of extracted content
            query_vector = self.tfidf_vectorizer.transform([extracted_content])
            
            # Step 2: Retrieve relevant interactions from user's own interactions
            user_relevant = self._get_user_relevant_interactions(
                user_id, query_vector, top_k // 2
            )
            
            # Step 3: Retrieve relevant interactions from all users' interactions
            global_relevant = self._get_global_relevant_interactions(
                user_id, query_vector, top_k // 2
            )
            
            return {
                "user_interactions": user_relevant,
                "global_interactions": global_relevant
            }
            
        except Exception as e:
            self.logger.warning(f"Error retrieving interactions: {e}")
            return {"user_interactions": [], "global_interactions": []}
    
    def _extract_content_from_query(self, query: str, task_type: str) -> str:
        """Extract relevant content from query based on task type."""
        query_lower = query.lower()
        
        if task_type == "news":
            # Extract content after "article:"
            if "article:" in query_lower:
                return query.split("article:")[-1].strip()
        elif task_type == "movie":
            # Extract content after "description:"
            if "description:" in query_lower:
                return query.split("description:")[-1].strip()
        elif task_type == "product_rating":
            # Extract content after "review:"
            if "review:" in query_lower:
                return query.split("review:")[-1].strip()
        
        # Fallback: return the query as is
        return query
    
    def _get_user_relevant_interactions(self, 
                                      user_id: str, 
                                      query_vector: np.ndarray, 
                                      top_k: int) -> List[Tuple[Interaction, float]]:
        """Get relevant interactions from user's own interactions."""
        if user_id not in self.user_interactions:
            return []
        
        user_interaction_ids = self.user_interactions[user_id]
        if not user_interaction_ids:
            return []
        
        user_results = []
        user_interaction_indices = []
        user_texts = []
        
        # Collect user's interaction texts and indices
        for node_id in user_interaction_ids:
            if node_id in self.interaction_ids:
                idx = self.interaction_ids.index(node_id)
                user_interaction_indices.append(idx)
                user_texts.append(self.interaction_texts[idx])
        
        if not user_texts:
            return []
        
        # Calculate similarities for user's interactions
        interaction_vectors = self.tfidf_vectorizer.transform(user_texts)
        similarities = cosine_similarity(query_vector, interaction_vectors).flatten()
        
        # Get top-k most similar interactions from user
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                original_idx = user_interaction_indices[idx]
                node_id = self.interaction_ids[original_idx]
                node = self.nodes[node_id]
                
                # Reconstruct interaction object
                interaction = Interaction(
                    id=node.properties["interaction_id"],
                    text=node.properties["text"],
                    title=node.properties["title"],
                    category=node.properties["category"],
                    timestamp=datetime.fromisoformat(node.properties["timestamp"])
                )
                
                user_results.append((interaction, similarities[idx]))
        
        return user_results
    
    def _get_global_relevant_interactions(self, 
                                        user_id: str, 
                                        query_vector: np.ndarray, 
                                        top_k: int) -> List[Tuple[Interaction, float]]:
        """Get relevant interactions from all users' interactions."""
        if not self.interaction_texts:
            return []
        
        # Calculate similarities for all interactions
        all_interaction_vectors = self.tfidf_vectorizer.transform(self.interaction_texts)
        similarities = cosine_similarity(query_vector, all_interaction_vectors).flatten()
        
        # Get top-k most similar interactions from all users
        top_indices = np.argsort(similarities)[::-1]
        
        global_results = []
        added_count = 0
        
        for idx in top_indices:
            if added_count >= top_k:
                break
                
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                node_id = self.interaction_ids[idx]
                node = self.nodes[node_id]
                
                # Skip if this is from the same user (to avoid duplicates)
                if node.properties.get("user_id") == user_id:
                    continue
                
                # Reconstruct interaction object
                interaction = Interaction(
                    id=node.properties["interaction_id"],
                    text=node.properties["text"],
                    title=node.properties["title"],
                    category=node.properties["category"],
                    timestamp=datetime.fromisoformat(node.properties["timestamp"])
                )
                
                global_results.append((interaction, similarities[idx]))
                added_count += 1
        
        return global_results
    
    def get_category_relationships(self, user_id: str) -> Dict[str, float]:
        """Get user's category preferences based on graph relationships."""
        if user_id not in self.user_interactions:
            return {}
        
        category_counts = defaultdict(int)
        total_interactions = len(self.user_interactions[user_id])
        
        for node_id in self.user_interactions[user_id]:
            if node_id in self.nodes:
                category = self.nodes[node_id].properties.get("category")
                if category:
                    category_counts[category] += 1
        
        # Convert to probabilities
        return {cat: count / total_interactions 
                for cat, count in category_counts.items()}
    
    def get_semantic_context(self, user_id: str, query: str, task_type: str = "news") -> Dict[str, Any]:
        """Get enhanced semantic context from the graph for personalized inference."""
        # Get both user and global relevant interactions
        relevant_interactions_dict = self.retrieve_relevant_interactions(user_id, query, task_type, top_k=6)
        user_interactions = relevant_interactions_dict["user_interactions"]
        global_interactions = relevant_interactions_dict["global_interactions"]
        
        category_prefs = self.get_category_relationships(user_id)
        
        # Extract concepts from relevant interactions
        relevant_concepts = set()
        for interaction, score in user_interactions + global_interactions:
            concepts = self._extract_entities(interaction.text + " " + interaction.title)
            relevant_concepts.update(concepts)
        
        return {
            "user_interactions": [
                {
                    "title": interaction.text,  # Use full text as title
                    "text": interaction.text,  # Use full text
                    "category": interaction.category,
                    "relevance_score": score
                }
                for interaction, score in user_interactions
            ],
            "global_interactions": [
                {
                    "title": interaction.text,  # Use full text as title
                    "text": interaction.text,  # Use full text
                    "category": interaction.category,
                    "relevance_score": score
                }
                for interaction, score in global_interactions
            ],
            "category_preferences": category_prefs,
            "relevant_concepts": list(relevant_concepts),
            "interaction_count": len(self.user_interactions.get(user_id, []))
        }


class PersonaAgent:
    """
    PersonaAgent that uses GraphRAG memory for personalized inference.
    """
    
    def __init__(self, 
                 categories: List[str],
                 model_name: str = "gpt-3.5-turbo",
                 persona_weight: float = 0.4,
                 task_type: str = "news",
                 task_config: Dict[str, str] = None):
        """
        Initialize the PersonaAgent.
        
        Args:
            categories: List of available categories
            model_name: LLM model to use for inference
            persona_weight: Weight for persona influence (0-1)
            task_type: Type of task ('news', 'movie', 'custom')
            task_config: Custom task configuration for prompts
        """
        self.categories = categories
        self.model_name = model_name
        self.persona_weight = persona_weight
        self.task_type = task_type
        self.memory = GraphRAGMemory()
        
        # Set task configuration
        self.task_config = task_config or self._get_default_task_config(task_type)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
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
                "instruction": "Based on the article content and the user's interaction history, classify this article.\nFirst consider the article content and also take the user's demonstrated preferences and relevant interactions and concepts as references.\n\nAnswer with just the category name."
            },
            "movie": {
                "content_type": "movie",
                "classification_term": "tags",
                "content_key": "description:",
                "category_term": "Tag", 
                "preference_term": "tag preferences",
                "interaction_type": "movies",
                "instruction": "Based on the movie description and the user's interaction history, classify this movie.\nFirst consider the movie description content and also take the user's demonstrated preferences and relevant interactions and concepts as references.\n\nAnswer with just the tag name."
            },
            "product_rating": {
                "content_type": "product review",
                "classification_term": "ratings",
                "content_key": "review:",
                "category_term": "Rating",
                "preference_term": "rating preferences",
                "interaction_type": "product reviews",
                "instruction": "Based on the product review text and the user's rating history, predict the rating (1-5 stars).\nConsider the user's personal review patterns, similar reviews from other users, and relevant concepts as references. Give higher weight to reviews with higher similarity scores.\nAnswer with just the rating number (1, 2, 3, 4, or 5)."
            },
            "custom": {
                "content_type": "item",
                "classification_term": "categories",
                "content_key": "content:",
                "category_term": "Category",
                "preference_term": "category preferences", 
                "interaction_type": "items",
                "instruction": "Based on the content and the user's interaction history, classify this item.\nFirst consider the content and also take the user's demonstrated preferences and relevant interactions and concepts as references.\n\nAnswer with just the category name."
            }
        }
        return configs.get(task_type, configs["custom"])
    
    def add_user_interactions(self, user_id: str, interactions: List[Dict[str, Any]]):
        """Add user interactions to the GraphRAG memory."""
        for i, interaction_data in enumerate(interactions):
            interaction = Interaction(
                id=interaction_data.get('id', f"{user_id}_{i}"),
                text=interaction_data.get('text', ''),
                title=interaction_data.get('title', ''),
                category=interaction_data.get('category', 'unknown')
            )
            self.memory.add_interaction(user_id, interaction)
    
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
                # For product rating, always use the review text directly
                display_text = interaction['text']
                base_prompt += f"{i}. \"{display_text}\" ({self.task_config['category_term']}: {interaction['category']}) - Similarity: {score:.3f} ({confidence})\n"
        
        # Add global relevant interactions from other users (sorted by similarity score)
        if semantic_context.get("global_interactions"):
            global_interactions = semantic_context["global_interactions"][:3]  # Show top 3
            base_prompt += f"\nSimilar {self.task_config['interaction_type']} from other users with similar interests (ordered by relevance):\n"
            for i, interaction in enumerate(global_interactions, 1):
                score = interaction['relevance_score']
                confidence = "HIGH" if score > 0.5 else "MEDIUM" if score > 0.2 else "LOW"
                # For product rating, always use the review text directly
                display_text = interaction['text']
                base_prompt += f"{i}. \"{display_text}\" ({self.task_config['category_term']}: {interaction['category']}) - Similarity: {score:.3f} ({confidence})\n"
        
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
        
        enhanced_instruction = self.task_config['instruction'].replace(
            "relevant interactions and concepts as references",
            "user's personal interactions, similar interactions from other users, and relevant concepts as references. Give higher weight to interactions with higher similarity scores"
        )
        base_prompt += f"\n{enhanced_instruction}"
        
        return base_prompt

    
    def predict(self, user_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Make a personalized prediction using GraphRAG context and LLM inference.
        
        Args:
            user_id: User identifier
            query: Input query to classify
            
        Returns:
            Tuple of (prediction, metadata)
        """
        
        pass
    
    def batch_predict(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make batch predictions for multiple examples."""
        results = []
        
        for example in test_data:
            example_id = example['id']
            query = example['input']
            profile_data = example.get('profile', [])
            
            # Add user interactions to GraphRAG memory
            self.add_user_interactions(example_id, profile_data)
            
            # Make prediction
            prediction, metadata = self.predict(example_id, query)
            
            result = {
                'id': example_id,
                'prediction': prediction,
                'metadata': metadata
            }
            
            results.append(result)
            
            # Log progress
            if len(results) % 10 == 0:
                self.logger.info(f"Processed {len(results)} examples")
        
        return results
    
    def evaluate(self, predictions: List[Dict[str, Any]], 
                ground_truth: Dict[str, str]) -> Dict[str, float]:
        """Evaluate predictions against ground truth."""
        correct = 0
        total = 0
        category_stats = {cat: {'correct': 0, 'total': 0, 'predicted': 0} 
                         for cat in self.categories}
        
        for pred in predictions:
            example_id = pred['id']
            predicted_category = pred['prediction']
            true_category = ground_truth.get(example_id)
            
            if true_category:
                total += 1
                if true_category in category_stats:
                    category_stats[true_category]['total'] += 1
                if predicted_category in category_stats:
                    category_stats[predicted_category]['predicted'] += 1
                
                if predicted_category == true_category:
                    correct += 1
                    if true_category in category_stats:
                        category_stats[true_category]['correct'] += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'total_examples': total,
            'correct_predictions': correct,
            'category_stats': category_stats
        }


def load_data(train_file: str, label_file: str) -> Tuple[List[Dict], Dict[str, str]]:
    """Load training data and labels."""
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    with open(label_file, 'r') as f:
        label_data = json.load(f)
    
    labels = {item['id']: item['output'] for item in label_data['golds']}
    return train_data, labels


if __name__ == "__main__":
    # Define categories
    categories = [
        "women", "religion", "politics", "style & beauty", "entertainment", 
        "culture & arts", "sports", "science & technology", "travel", 
        "business", "crime", "education", "healthy living", "parents", "food & drink"
    ]
    
    # Initialize LLM PersonaAgent with GraphRAG
    agent = PersonaAgent(categories=categories, persona_weight=0.4)
    
    print("Loading data...")
    train_data, labels = load_data('../news_train.json', '../news_label.json')
    
    print(f"Loaded {len(train_data)} examples with {len(labels)} labels")
    print(f"Categories: {categories}")
    
    # Make predictions
    print("\nMaking predictions with GraphRAG-enhanced PersonaAgent...")
    predictions = agent.batch_predict(train_data)
    
    # Evaluate
    evaluation = agent.evaluate(predictions, labels)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Correct: {evaluation['correct_predictions']}/{evaluation['total_examples']}")
    
    # Show some example predictions
    print(f"\n=== Example Predictions ===")
    for i, pred in enumerate(predictions[:3]):
        example_id = pred['id']
        predicted = pred['prediction']
        actual = labels.get(example_id, 'Unknown')
        print(f"\nExample {example_id}:")
        print(f"  Predicted: {predicted}")
        print(f"  Actual: {actual}")
        print(f"  Correct: {'✓' if predicted == actual else '✗'}")
        print(f"  User interactions: {pred['metadata']['user_interactions']}")
