"""
Configurable PersonaAgent Demo - allows specification of datasets, models, and parameters.
"""

import sys
import os
import argparse
import json
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from persona_agent import PersonaAgent
from llm_integration import OllamaPersonaAgent, AdaptiveOllamaPersonaAgent
from news_categorization_dataset_config import get_dataset_config, list_available_datasets, DatasetConfig


class PersonaAgentRunner:
    """Configurable PersonaAgent runner supporting multiple datasets and models."""
    
    def __init__(self, 
                 dataset_name: str = 'news',
                 model_name: str = 'llama3',
                 ollama_host: str = 'http://localhost:11434',
                 persona_weight: float = 0.4,
                 sample_size: int = None,
                 custom_config: Dict[str, Any] = None):
        """
        Initialize the configurable PersonaAgent runner.
        
        Args:
            dataset_name: Name of the dataset to use ('news', 'movies', 'movies_full')
            model_name: LLM model name for Ollama
            ollama_host: Ollama server host
            persona_weight: Weight for persona influence (0-1)
            sample_size: Number of examples to use (None for all)
            custom_config: Custom dataset configuration
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.persona_weight = persona_weight
        self.sample_size = sample_size
        self.custom_config = custom_config
        
        # Load dataset configuration
        if custom_config:
            self.config = DatasetConfig(**custom_config)
        else:
            self.config = get_dataset_config(dataset_name)
        
        # Initialize agent with task type
        task_type = "movie" if "movie" in dataset_name.lower() else "news"
        self.agent = AdaptiveOllamaPersonaAgent(
            categories=self.config.categories,
            model_name=model_name,
            ollama_host=ollama_host,
            persona_weight=persona_weight,
            task_type=task_type
        )
        
        self.train_data = []  # Training data for building GraphRAG memory
        self.test_data = []   # Test data for evaluation
        self.labels = {}
    
    def load_data(self):
        """Load and prepare the dataset."""
        print(f"📁 Loading {self.config.name} dataset...")
        try:
            self.train_data, self.test_data, self.labels = self.config.load_data()
            
            # Apply sample size limit to test data if specified
            if self.sample_size and self.sample_size < len(self.test_data):
                self.test_data = self.test_data[:self.sample_size]
                print(f"   📊 Using sample of {self.sample_size} test examples")
            
            # Count interactions in training data (for GraphRAG memory)
            train_interactions = 0
            for user_data in self.train_data:
                profile = user_data.get('profile', [])
                train_interactions += len(profile)
            
            # Count queries in test data (for evaluation)
            test_queries = 0
            for user_data in self.test_data:
                queries = user_data.get('query', [])
                test_queries += len(queries)
            
            print(f"   ✓ Training data: {len(self.train_data)} users with {train_interactions} interactions")
            print(f"   ✓ Test data: {len(self.test_data)} users with {test_queries} queries")
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            raise
    
    def build_memory(self):
        """Build GraphRAG memory from both training and test data."""
        print(f"🧠 Building GraphRAG memory from both training and test data...")
        
        # Add training data users to memory
        train_users_added = 0
        for example in self.train_data:
            # Get user ID (handle different formats)
            user_id = str(example.get('id', example.get('user_id', 'unknown')))
            
            # Convert profile to standard format
            profile = example.get('profile', [])
            standard_profile = self.config.convert_profile_to_standard(profile)
            
            # Add to agent memory
            self.agent.add_user_interactions(user_id, standard_profile)
            train_users_added += 1
        
        # Add test data users to memory (for their profiles)
        test_users_added = 0
        for example in self.test_data:
            # Get user ID (handle different formats)
            user_id = str(example.get('id', example.get('user_id', 'unknown')))
            
            # Convert profile to standard format
            profile = example.get('profile', [])
            standard_profile = self.config.convert_profile_to_standard(profile)
            
            # Add to agent memory
            self.agent.add_user_interactions(user_id, standard_profile)
            test_users_added += 1
        
        # Calculate total interactions
        total_interactions = sum(len(interactions) for interactions in self.agent.memory.user_interactions.values())
        print(f"   ✓ Added {train_users_added} training users to GraphRAG memory")
        print(f"   ✓ Added {test_users_added} test users to GraphRAG memory")
        print(f"   ✓ Total users in memory: {len(self.agent.memory.user_interactions)}")
        print(f"   ✓ Total interactions in memory: {total_interactions}")
    
    def analyze_dataset(self):
        """Analyze and display dataset statistics."""
        print(f"\n📊 {self.config.name} Dataset Analysis:")
        print(f"   Training users: {len(self.train_data)}")
        print(f"   Test users: {len(self.test_data)}")
        print(f"   Categories: {len(self.config.categories)}")
        
        # Analyze training data (for memory building)
        train_interactions = 0
        for user_data in self.train_data:
            profile = user_data.get('profile', [])
            train_interactions += len(profile)
        
        # Analyze test data (for evaluation)
        test_queries = 0
        label_counts = {}
        
        for user_data in self.test_data:
            queries = user_data.get('query', [])
            test_queries += len(queries)
            
            for query in queries:
                ground_truth = query.get('gold', 'unknown')
                label_counts[ground_truth] = label_counts.get(ground_truth, 0) + 1
        
        print(f"   Training interactions: {train_interactions}")
        print(f"   Test queries: {test_queries}")
        
        # Show category distribution in labels
        if label_counts:
            top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top test labels: {dict(top_labels)}")
        
        # Show training data statistics
        if self.train_data:
            interaction_counts = [len(user_data.get('profile', [])) for user_data in self.train_data]
            if interaction_counts:
                avg_interactions = sum(interaction_counts) / len(interaction_counts)
                print(f"   Avg training interactions per user: {avg_interactions:.1f}")
        
        # Show test data statistics
        if self.test_data:
            query_counts = [len(user_data.get('query', [])) for user_data in self.test_data]
            if query_counts:
                avg_queries = sum(query_counts) / len(query_counts)
                print(f"   Avg test queries per user: {avg_queries:.1f}")
    
    def demonstrate_predictions(self, num_examples: int = 3):
        """Demonstrate predictions on sample test examples."""
        print(f"\n🎯 Prediction Demonstrations (using test data):")
        
        try:
            demo_count = 0
            for user_data in self.test_data:
                if demo_count >= num_examples:
                    break
                    
                user_id = str(user_data.get('user_id', 'unknown'))
                queries = user_data.get('query', [])
                
                # Take first query from this user for demonstration
                if queries:
                    query_data = queries[0]
                    query_id = str(query_data.get('id', 'unknown'))
                    query_input = query_data.get('input', '')
                    ground_truth = query_data.get('gold', 'Unknown')
                    
                    # Make prediction
                    prediction, metadata = self.agent.predict(user_id, query_input)
                    
                    demo_count += 1
                    print(f"\n   Example {demo_count}:")
                    print(f"   Test User ID: {user_id}")
                    print(f"   Query ID: {query_id}")
                    print(f"   Query: {query_input[:100]}{'...' if len(query_input) > 100 else ''}")
                    print(f"   Prediction: {prediction}")
                    print(f"   True label: {ground_truth}")
                    print(f"   Correct: {'✓' if prediction == ground_truth else '✗'}")
                    
                    # Show global retrieval effectiveness
                    semantic_context = metadata.get('semantic_context', {})
                    user_interactions = semantic_context.get('user_interactions', [])
                    global_interactions = semantic_context.get('global_interactions', [])
                    
                    print(f"   Retrieved from training data: {len(global_interactions)} interactions")
                    print(f"   User's own interactions: {len(user_interactions)} interactions")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise
    
    def run_evaluation(self):
        """Run optimized evaluation on the dataset."""
        print(f"\n📈 Running Optimized Evaluation:")
        
        # Prepare evaluation data - collect all queries from all test users
        evaluation_queries = []
        evaluation_labels = {}
        
        for user_data in self.test_data:
            user_id = str(user_data.get('user_id', 'unknown'))
            queries = user_data.get('query', [])
            
            for query in queries:
                query_id = str(query.get('id', 'unknown'))
                query_input = query.get('input', '')
                ground_truth = query.get('gold', 'unknown')
                
                # Create evaluation item
                eval_item = {
                    'user_id': user_id,
                    'id': query_id,
                    'input': query_input
                }
                evaluation_queries.append(eval_item)
                evaluation_labels[query_id] = ground_truth
        
        print(f"   Total queries to evaluate: {len(evaluation_queries)}")
        
        # Use optimized batch prediction
        predictions = self._run_batch_predictions(evaluation_queries)
        
        # Calculate accuracy
        correct = 0
        total = len(evaluation_queries)
        
        for eval_item in evaluation_queries:
            query_id = eval_item['id']
            ground_truth = evaluation_labels[query_id]
            prediction = predictions.get(query_id, 'unknown')
            
            if prediction == ground_truth:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Correct: {correct}/{total}")
        
        # Calculate F1 scores for all categories
        f1_scores = self._calculate_f1_scores(predictions, evaluation_labels)
        
        # Calculate per-category performance
        category_stats = {}
        for query_id, ground_truth in evaluation_labels.items():
            if ground_truth not in category_stats:
                category_stats[ground_truth] = {'total': 0, 'correct': 0, 'predicted': 0}
            category_stats[ground_truth]['total'] += 1
            
            prediction = predictions.get(query_id, 'unknown')
            if prediction == ground_truth:
                category_stats[ground_truth]['correct'] += 1
            
            # Count predictions for this category (for precision calculation)
            if prediction not in category_stats:
                category_stats[prediction] = {'total': 0, 'correct': 0, 'predicted': 0}
            category_stats[prediction]['predicted'] += 1
        
        # Show per-category performance for top categories
        performing_categories = [(cat, stats) for cat, stats in category_stats.items() 
                               if stats['total'] > 0]
        performing_categories.sort(key=lambda x: x[1]['total'], reverse=True)
        
        print(f"\n   📊 Overall Performance:")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     Macro F1: {f1_scores['macro']['f1']:.4f}")
        print(f"     Weighted F1: {f1_scores['weighted']['f1']:.4f}")
        
        print(f"\n   📊 Top Category Performance:")
        for cat, stats in performing_categories[:5]:
            precision = stats['correct'] / stats['predicted'] if stats['predicted'] > 0 else 0
            recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            f1_info = f1_scores['per_category'].get(cat, {})
            f1_score = f1_info.get('f1', 0)
            print(f"     {cat}: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}, Support={stats['total']}")
        
        # Create evaluation result
        evaluation = {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_examples': total,
            'f1_scores': f1_scores,
            'category_stats': category_stats,
            'predictions': predictions,
            'labels': evaluation_labels
        }
        
        # Save results to file
        saved_file = self._save_evaluation_results(evaluation)
        print(f"\n   💾 Results saved to: {saved_file}")
        
        return evaluation
    
    def _run_batch_predictions(self, evaluation_queries):
        """Run optimized batch predictions with caching and progress tracking."""
        import concurrent.futures
        import time
        from collections import defaultdict
        
        predictions = {}
        context_cache = {}  # Cache semantic contexts for similar users
        
        # Group queries by user for better caching
        user_queries = defaultdict(list)
        for eval_item in evaluation_queries:
            user_queries[eval_item['user_id']].append(eval_item)
        
        print(f"   Grouped into {len(user_queries)} users")
        print(f"   Starting batch prediction with caching...")
        
        start_time = time.time()
        processed = 0
        total_queries = len(evaluation_queries)
        
        # Process queries user by user for better context reuse
        for user_id, queries in user_queries.items():
            print(f"   Processing user {user_id}: {len(queries)} queries", end=" -> ")
            
            # Pre-compute semantic context for this user (if not cached)
            if user_id not in context_cache:
                # Get a sample query to build initial context
                sample_query = queries[0]['input']
                semantic_context = self.agent.memory.get_semantic_context(
                    user_id, sample_query, task_type=getattr(self.agent, 'task_type', 'news')
                )
                context_cache[user_id] = semantic_context
            
            # Use cached context
            cached_context = context_cache[user_id]
            
            # Process queries for this user in smaller batches
            batch_size = 10  # Process 10 queries at a time per user
            user_predictions = {}
            
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                
                # Process batch with threading for I/O bound LLM calls
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_query = {}
                    
                    for eval_item in batch:
                        query_id = eval_item['id']
                        query_input = eval_item['input']
                        
                        # Submit prediction task
                        future = executor.submit(
                            self._predict_with_cached_context, 
                            user_id, query_input, cached_context
                        )
                        future_to_query[future] = query_id
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(future_to_query):
                        query_id = future_to_query[future]
                        try:
                            prediction = future.result(timeout=30)  # 30 second timeout
                            user_predictions[query_id] = prediction
                        except Exception as e:
                            print(f"Error predicting query {query_id}: {e}")
                            user_predictions[query_id] = 'entertainment'  # Default fallback
                
                processed += len(batch)
                
                # Progress update
                if processed % 50 == 0 or processed == total_queries:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total_queries - processed) / rate if rate > 0 else 0
                    print(f"\r   Progress: {processed}/{total_queries} ({processed/total_queries*100:.1f}%) - "
                          f"{rate:.1f} queries/sec - ETA: {eta:.1f}s", end="")
            
            predictions.update(user_predictions)
            print(f" ✓ {len(user_predictions)} predictions")
        
        total_time = time.time() - start_time
        rate = len(evaluation_queries) / total_time if total_time > 0 else 0  
        print(f"\n   ✓ Completed {len(predictions)} predictions in {total_time:.1f}s ({rate:.1f} queries/sec)")
        
        return predictions
    
    def _predict_with_cached_context(self, user_id, query_input, cached_context):
        """Make prediction using cached semantic context to avoid redundant computation."""
        # Generate personalized prompt using cached context
        personalized_prompt = self.agent._generate_personalized_prompt(query_input, cached_context)
        
        # Call LLM directly to avoid recomputing context
        if hasattr(self.agent, '_call_ollama_model'):
            prediction = self.agent._call_ollama_model(personalized_prompt)
        else:
            # Fallback to standard prediction
            prediction, _ = self.agent.predict(user_id, query_input)
        
        # Ensure prediction is valid
        if prediction not in self.agent.categories:
            if cached_context.get("category_preferences"):
                prediction = max(cached_context["category_preferences"].items(), 
                               key=lambda x: x[1])[0]
            else:
                prediction = "entertainment"
        
        return prediction
    
    def _calculate_f1_scores(self, predictions: Dict[str, str], labels: Dict[str, str]) -> Dict[str, Any]:
        """Calculate F1 scores for all categories."""
        # Get all categories that appear in either predictions or labels
        all_categories = set(list(predictions.values()) + list(labels.values()))
        
        f1_scores = {}
        macro_precision_sum = 0
        macro_recall_sum = 0
        macro_f1_sum = 0
        valid_categories = 0
        
        for category in all_categories:
            if category == 'unknown':
                continue
                
            # Calculate TP, FP, FN for this category
            tp = sum(1 for query_id in predictions.keys() 
                    if predictions[query_id] == category and labels.get(query_id) == category)
            
            fp = sum(1 for query_id in predictions.keys() 
                    if predictions[query_id] == category and labels.get(query_id) != category)
            
            fn = sum(1 for query_id in labels.keys() 
                    if labels[query_id] == category and predictions.get(query_id) != category)
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            support = sum(1 for label in labels.values() if label == category)
            
            f1_scores[category] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            if support > 0:  # Only include categories that actually exist in labels
                macro_precision_sum += precision
                macro_recall_sum += recall
                macro_f1_sum += f1
                valid_categories += 1
        
        # Calculate macro averages
        macro_precision = macro_precision_sum / valid_categories if valid_categories > 0 else 0
        macro_recall = macro_recall_sum / valid_categories if valid_categories > 0 else 0
        macro_f1 = macro_f1_sum / valid_categories if valid_categories > 0 else 0
        
        # Calculate weighted averages
        total_support = sum(scores['support'] for scores in f1_scores.values())
        if total_support > 0:
            weighted_precision = sum(scores['precision'] * scores['support'] for scores in f1_scores.values()) / total_support
            weighted_recall = sum(scores['recall'] * scores['support'] for scores in f1_scores.values()) / total_support
            weighted_f1 = sum(scores['f1'] * scores['support'] for scores in f1_scores.values()) / total_support
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0
        
        return {
            'per_category': f1_scores,
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            },
            'weighted': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            }
        }
    
    def _save_evaluation_results(self, evaluation: Dict[str, Any], filename: str = None) -> str:
        """Save evaluation results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{self.dataset_name}_{self.model_name}_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'experiment_info': {
                'dataset': self.dataset_name,
                'model': self.model_name,
                'persona_weight': self.persona_weight,
                'sample_size': self.sample_size,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'train_file': getattr(self.config, 'train_file', 'N/A'),
                    'test_file': getattr(self.config, 'test_file', 'N/A'),
                    'categories': self.config.categories
                }
            },
            'metrics': {
                'accuracy': evaluation.get('accuracy', 0),
                'correct_predictions': evaluation.get('correct_predictions', 0),
                'total_examples': evaluation.get('total_examples', 0),
                'f1_scores': evaluation.get('f1_scores', {}),
                'category_stats': evaluation.get('category_stats', {})
            },
            'predictions': evaluation.get('predictions', {}),
            'labels': evaluation.get('labels', {})
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        return filepath
        
    def run_complete_demo(self):
        """Run the complete demo pipeline."""
        print("="*60)
        print(f" PERSONAAGENT CONFIGURABLE DEMO")
        print("="*60)
        print(f"🤖 Dataset: {self.config.name}")
        print(f"🧠 Model: {self.model_name}")
        print(f"⚙️  Configuration: {self.dataset_name}")
        
        # Load and analyze data
        self.load_data()
        self.analyze_dataset()
        
        # Build memory
        self.build_memory()
        
        # Demonstrate predictions
        self.demonstrate_predictions()
        
        evaluation = self.run_evaluation()
        
        print("\n" + "="*60)
        print(" DEMO COMPLETE")
        print("="*60)
        print(f"🎉 {self.config.name} demo completed!")
        print(f"   Final accuracy: {evaluation['accuracy']:.4f}")
        
        return evaluation


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Configurable PersonaAgent Demo')
    
    parser.add_argument('--dataset', '-d', 
                       choices=list_available_datasets(), 
                       default='news',
                       help='Dataset to use for demo')
    
    parser.add_argument('--model', '-m',
                       default='llama3',
                       help='Model name for Ollama')
    
    parser.add_argument('--host',
                       default='http://localhost:11434',
                       help='Ollama server host')
    
    parser.add_argument('--sample-size', '-s',
                       type=int,
                       help='Number of examples to use (default: all)')
    
    parser.add_argument('--persona-weight', '-w',
                       type=float,
                       default=0.4,
                       help='Persona weight (0-1)')
    
    parser.add_argument('--config-file',
                       help='JSON file with custom dataset configuration')
    
    parser.add_argument('--list-datasets', '-l',
                       action='store_true',
                       help='List available datasets and exit')
    
    return parser.parse_args()


def load_custom_config(config_file: str) -> Dict[str, Any]:
    """Load custom configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    """Main function with argument parsing."""
    args = parse_arguments()
    
    # Handle list datasets command
    if args.list_datasets:
        print("Available datasets:")
        for dataset in list_available_datasets():
            config = get_dataset_config(dataset)
            print(f"  {dataset}: {config.name} ({len(config.categories)} categories)")
        return
    
    # Load custom configuration if specified
    custom_config = None
    if args.config_file:
        try:
            custom_config = load_custom_config(args.config_file)
            print(f"📋 Loaded custom configuration from {args.config_file}")
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            return
    
    # Initialize and run demo
    try:
        runner = PersonaAgentRunner(
            dataset_name=args.dataset,
            model_name=args.model,
            ollama_host=args.host,
            persona_weight=args.persona_weight,
            sample_size=args.sample_size,
            custom_config=custom_config
        )
        
        runner.run_complete_demo()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
