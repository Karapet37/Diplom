"""
Dataset configuration for PersonaAgent supporting multiple domains.
"""

from typing import Dict, List, Any, Tuple
import json


class DatasetConfig:
    """Configuration for different datasets."""
    
    def __init__(self, name: str, categories: List[str], 
                 train_file: str = None, test_file: str = None, label_file: str = None,
                 profile_format: Dict[str, str] = None):
        self.name = name
        self.categories = categories
        self.train_file = train_file  # Training data for building GraphRAG memory
        self.test_file = test_file or train_file  # Test data for evaluation (defaults to train_file for backward compatibility)
        self.label_file = label_file
        self.profile_format = profile_format or {
            'title_key': 'title',
            'text_key': 'text', 
            'category_key': 'category'
        }
    
    def load_data(self) -> Tuple[List[Dict], List[Dict], Dict[str, str]]:
        """Load training data, test data, and labels for this dataset."""
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
                    ground_truth = query.get('gold', 'unknown')
                    labels[query_id] = ground_truth
        
        return train_data, test_data, labels
    
    def convert_profile_to_standard(self, profile: List[Dict]) -> List[Dict]:
        """Convert dataset-specific profile format to standard format."""
        standard_profile = []
        title_key = self.profile_format['title_key']
        text_key = self.profile_format['text_key']
        category_key = self.profile_format['category_key']
        
        for interaction in profile:
            # Handle different formats
            if title_key == 'tag' and text_key == 'description':
                # Movie format
                standard_interaction = {
                    'id': interaction.get('id', ''),
                    'title': f"Movie: {interaction.get(title_key, 'Unknown')}",
                    'text': interaction.get(text_key, ''),
                    'category': interaction.get(category_key, 'unknown')
                }
            else:
                # News format or standard format
                standard_interaction = {
                    'id': interaction.get('id', ''),
                    'title': interaction.get(title_key, ''),
                    'text': interaction.get(text_key, ''),
                    'category': interaction.get(category_key, 'unknown')
                }
            standard_profile.append(standard_interaction)
        return standard_profile


# Predefined Dataset Configurations
DATASET_CONFIGS = {
    'news': DatasetConfig(
        name='News Articles',
        categories=[
            "women", "religion", "politics", "style & beauty", "entertainment", 
            "culture & arts", "sports", "science & technology", "travel", 
            "business", "crime", "education", "healthy living", "parents", "food & drink"
        ],
        train_file='data/news/user_others.json',  # Training data for GraphRAG memory
        test_file='data/news/user_top_100_history.json',  # Test data for evaluation
        label_file=None, # Labels are embedded in the data
        profile_format={
            'title_key': 'title',
            'text_key': 'text',
            'category_key': 'category'
        }
    ),
    'news_small': DatasetConfig(
        name='News Articles (Small)',
        categories=[
            "women", "religion", "politics", "style & beauty", "entertainment", 
            "culture & arts", "sports", "science & technology", "travel", 
            "business", "crime", "education", "healthy living", "parents", "food & drink"
        ],
        train_file='data/news/user_others_small.json',  # Training data for GraphRAG memory
        test_file='data/news/user_top_100_history_small.json',  # Test data for evaluation
        label_file=None,  # Labels are embedded in the test data
        profile_format={
            'title_key': 'title',
            'text_key': 'text',
            'category_key': 'category'
        }
    ),
    
    'movies': DatasetConfig(
        name='Movie Descriptions',
        categories=[
            "sci-fi", "based on a book", "comedy", "action", "twist ending", 
            "dystopia", "dark comedy", "classic", "psychology", "fantasy", 
            "romance", "thought-provoking", "social commentary", "violence", "true story"
        ],
        train_file='data/movies/user_others.json',
        test_file='data/movies/user_top_100_history.json',  # Test data for evaluation
        label_file=None,  # Labels are embedded in the test data
        profile_format={
            'title_key': 'tag',
            'text_key': 'description',
            'category_key': 'tag'
        }
    )
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASET_CONFIGS:
        available = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    return DATASET_CONFIGS[dataset_name]


def list_available_datasets() -> List[str]:
    """List all available dataset configurations."""
    return list(DATASET_CONFIGS.keys())
