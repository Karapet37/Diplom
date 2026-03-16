"""
Product Rating dataset configuration for PersonaAgent.
"""

from typing import Dict, List, Any, Tuple
import json


class ProductRatingDatasetConfig:
    """Configuration for product rating datasets."""
    
    def __init__(self, name: str, categories: List[str], 
                 train_file: str = None, test_file: str = None, label_file: str = None,
                 profile_format: Dict[str, str] = None):
        self.name = name
        self.categories = categories
        self.train_file = train_file  # Training data for building GraphRAG memory
        self.test_file = test_file or train_file  # Test data for evaluation
        self.label_file = label_file
        self.profile_format = profile_format or {
            'title_key': 'title',
            'text_key': 'text', 
            'category_key': 'score'
        }
    
    def load_data(self) -> Tuple[List[Dict], List[Dict], Dict[str, str]]:
        """Load training data, test data, and labels for product rating dataset."""
        # Load training data for building GraphRAG memory
        with open(self.train_file, 'r') as f:
            train_data = json.load(f)
        
        # Load test data for evaluation
        with open(self.test_file, 'r') as f:
            test_data = json.load(f)
        
        # Handle different label formats - labels come from test data
        if self.label_file:
            # Traditional format with separate label file
            with open(self.label_file, 'r') as f:
                label_data = json.load(f)
            labels = {item['id']: item['output'] for item in label_data['golds']}
        else:
            # New format with embedded labels in test data queries
            labels = {}
            for user_data in test_data:
                queries = user_data.get('query', [])
                for query in queries:
                    query_id = str(query.get('id', 'unknown'))
                    ground_truth = str(query.get('gold', 'unknown'))
                    labels[query_id] = ground_truth
        
        return train_data, test_data, labels
    
    def convert_profile_to_standard(self, profile: List[Dict]) -> List[Dict]:
        """Convert product rating profile format to standard format."""
        standard_profile = []
        title_key = self.profile_format['title_key']
        text_key = self.profile_format['text_key']
        category_key = self.profile_format['category_key']
        
        for interaction in profile:
            # Get the review text
            review_text = interaction.get(text_key, '')
            
            # Create a meaningful title from the review text (first 50 chars)
            # title = review_text[:50] + "..." if len(review_text) > 50 else review_text
            title = review_text
            
            # Product rating format
            standard_interaction = {
                'id': interaction.get('id', ''),
                'title': title,  # Use truncated review text as title
                'text': review_text,  # Full review text
                'category': str(interaction.get(category_key, 'unknown'))
            }
            standard_profile.append(standard_interaction)
        return standard_profile


# Predefined Product Rating Dataset Configurations
PRODUCT_RATING_DATASET_CONFIGS = {
    'product_rating': ProductRatingDatasetConfig(
        name='Product Ratings',
        categories=["1", "2", "3", "4", "5"],  # 1-5 star ratings
        train_file='data/product_rating/user_others_sampled_10_19899.json', #'data/product_rating/user_others.json',  # Training data for GraphRAG memory
        test_file='data/product_rating/user_others_sampled_5_19899.json', #user_top_100_history.json', #user_top_100_history.json', #user_others.json', #'data/product_rating/user_top_100_history.json',  # Test data for evaluation
        label_file=None,  # Labels are embedded in the test data
        profile_format={
            'title_key': 'title',
            'text_key': 'text',
            'category_key': 'score'
        }
    ),
    'product_rating_small': ProductRatingDatasetConfig(
        name='Product Ratings (Small)',
        categories=["1", "2", "3", "4", "5"],  # 1-5 star ratings
        train_file='data/product_rating/user_top_100_history_max50.json', #user_others_sampled_100_19899.json', #user_others_small.json',  # Training data for GraphRAG memory
        test_file='data/product_rating/user_others_sampled_50.json', #user_top_100_history.json', #user_top_100_history.json', #user_top_100_history.json', #user_others.json', #user_top_100_history_small.json',  # Test data for evaluation
        label_file=None,  # Labels are embedded in the test data
        profile_format={
            'title_key': 'title',
            'text_key': 'text',
            'category_key': 'score'
        }
    ),
}


def get_product_rating_dataset_config(dataset_name: str) -> ProductRatingDatasetConfig:
    """Get configuration for a specific product rating dataset."""
    if dataset_name not in PRODUCT_RATING_DATASET_CONFIGS:
        available = list(PRODUCT_RATING_DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown product rating dataset '{dataset_name}'. Available: {available}")
    
    return PRODUCT_RATING_DATASET_CONFIGS[dataset_name]


def list_available_product_rating_datasets() -> List[str]:
    """List all available product rating dataset configurations."""
    return list(PRODUCT_RATING_DATASET_CONFIGS.keys())


def get_product_rating_task_config() -> Dict[str, str]:
    """Get task configuration for product rating prediction."""
    return {
        "content_type": "product review",
        "classification_term": "ratings",
        "content_key": "review:",
        "category_term": "Rating",
        "preference_term": "rating preferences",
        "interaction_type": "product reviews",
        "instruction": "Based on the product review text and the user's rating history, predict the rating (1-5 stars).\nConsider the user's personal review patterns, similar reviews from other users, and relevant concepts as references. Give higher weight to reviews with higher similarity scores.\nAnswer with just the rating number (1, 2, 3, 4, or 5)."
    }
