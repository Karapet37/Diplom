"""
Product Rating PersonaAgent Demo - predicts product ratings based on user review text.
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
from llm_integration import AdaptivePersonaAgent
from parallel_memory_builder import ParallelMemoryBuilder
from product_rating_dataset_config import (
    get_product_rating_dataset_config, 
    list_available_product_rating_datasets, 
    ProductRatingDatasetConfig,
    get_product_rating_task_config
)


class ProductRatingRunner:
    """Product Rating PersonaAgent runner."""
    
    def __init__(self, 
                 dataset_name: str = 'product_rating',
                 model_name: str = 'llama3',
                 ollama_host: str = 'http://localhost:11434',
                 persona_weight: float = 0.4,
                 sample_size: int = None,
                 custom_config: Dict[str, Any] = None):
        """
        Initialize the product rating PersonaAgent runner.
        
        Args:
            dataset_name: Name of the dataset to use ('product_rating', 'amazon_reviews', etc.)
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
            self.config = ProductRatingDatasetConfig(**custom_config)
        else:
            self.config = get_product_rating_dataset_config(dataset_name)
        
        # Initialize agent with product rating task type
        self.agent = AdaptivePersonaAgent(
            categories=self.config.categories,
            model_name=model_name,
            ollama_host=ollama_host,
            persona_weight=persona_weight,
            task_type="product_rating",
            task_config=get_product_rating_task_config()
        )
        
        self.train_data = []
        self.test_data = []
        self.labels = {}
    
    def load_data(self):
        """Load and prepare the product rating dataset."""
        print(f"📁 Loading {self.config.name} dataset...")
        try:
            self.train_data, self.test_data, self.labels = self.config.load_data()
            
            # Apply sample size limit to test data if specified
            if self.sample_size and self.sample_size < len(self.test_data):
                self.test_data = self.test_data[:self.sample_size]
                print(f"   📊 Using sample of {self.sample_size} test examples")
            
            # Count interactions in training data
            train_interactions = 0
            for user_data in self.train_data:
                profile = user_data.get('profile', [])
                train_interactions += len(profile)
            
            # Count queries in test data
            test_queries = 0
            for user_data in self.test_data:
                queries = user_data.get('query', [])
                test_queries += len(queries)
            
            print(f"   ✓ Training data: {len(self.train_data)} users with {train_interactions} product reviews")
            print(f"   ✓ Test data: {len(self.test_data)} users with {test_queries} rating queries")
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            raise
    
    def build_memory(self, use_parallel: bool = True, num_processes: int = None):
        """Build GraphRAG memory from training and test data."""
        print(f"🧠 Building GraphRAG memory for product rating prediction...")
        
        if use_parallel and len(self.train_data) + len(self.test_data) > 50:
            # Use parallel processing for larger datasets
            print(f"   🚀 Using parallel processing with {num_processes or 'auto'} processes...")
            self.build_memory_parallel(num_processes)
        else:
            # Use sequential processing for smaller datasets
            print(f"   📝 Using sequential processing...")
            self.build_memory_sequential()
    
    def build_memory_parallel(self, num_processes: int = None):
        """Build GraphRAG memory using parallel processing."""
        # Create parallel memory builder
        builder = ParallelMemoryBuilder(num_processes=num_processes)
        
        # Build memory in parallel
        memory = builder.build_memory_parallel(
            train_data=self.train_data,
            test_data=self.test_data,
            categories=self.config.categories
        )
        
        # Replace the agent's memory with the parallel-built memory
        self.agent.memory = memory
        
        total_interactions = sum(len(interactions) for interactions in memory.user_interactions.values())
        print(f"   ✓ Built memory with {len(memory.user_interactions)} users")
        print(f"   ✓ Total product reviews in memory: {total_interactions}")
        print(f"   ✓ Memory nodes: {len(memory.nodes)}")
        print(f"   ✓ Graph edges: {memory.graph.number_of_edges()}")
    
    def build_memory_sequential(self):
        """Build GraphRAG memory using sequential processing (original method)."""
        # Add training data users to memory
        train_users_added = 0
        for example in self.train_data:
            user_id = str(example.get('id', example.get('user_id', 'unknown')))
            profile = example.get('profile', [])
            standard_profile = self.config.convert_profile_to_standard(profile)
            self.agent.add_user_interactions(user_id, standard_profile)
            train_users_added += 1
        
        # Add test data users to memory
        test_users_added = 0
        for example in self.test_data:
            user_id = str(example.get('id', example.get('user_id', 'unknown')))
            profile = example.get('profile', [])
            standard_profile = self.config.convert_profile_to_standard(profile)
            self.agent.add_user_interactions(user_id, standard_profile)
            test_users_added += 1
        
        total_interactions = sum(len(interactions) for interactions in self.agent.memory.user_interactions.values())
        print(f"   ✓ Added {train_users_added} training users to GraphRAG memory")
        print(f"   ✓ Added {test_users_added} test users to GraphRAG memory")
        print(f"   ✓ Total users in memory: {len(self.agent.memory.user_interactions)}")
        print(f"   ✓ Total product reviews in memory: {total_interactions}")
    
    def analyze_dataset(self):
        """Analyze and display dataset statistics."""
        print(f"\n📊 {self.config.name} Dataset Analysis:")
        print(f"   Training users: {len(self.train_data)}")
        print(f"   Test users: {len(self.test_data)}")
        print(f"   Rating categories: {len(self.config.categories)} (1-5 stars)")
        
        # Analyze rating distribution in labels
        rating_counts = {}
        for rating in self.labels.values():
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        if rating_counts:
            print(f"   Rating distribution in test data:")
            for rating in sorted(rating_counts.keys()):
                count = rating_counts[rating]
                percentage = (count / len(self.labels)) * 100
                print(f"     {rating} stars: {count} reviews ({percentage:.1f}%)")
        
        # Show training data statistics
        if self.train_data:
            interaction_counts = [len(user_data.get('profile', [])) for user_data in self.train_data]
            if interaction_counts:
                avg_interactions = sum(interaction_counts) / len(interaction_counts)
                print(f"   Avg training reviews per user: {avg_interactions:.1f}")
    
    def demonstrate_predictions(self, num_examples: int = 3):
        """Demonstrate rating predictions on sample test examples."""
        print(f"\n🎯 Product Rating Prediction Demonstrations:")
        
        try:
            demo_count = 0
            for user_data in self.test_data:
                if demo_count >= num_examples:
                    break
                    
                user_id = str(user_data.get('user_id', 'unknown'))
                queries = user_data.get('query', [])
                
                if queries:
                    query_data = queries[0]
                    query_id = str(query_data.get('id', 'unknown'))
                    query_input = query_data.get('input', '')
                    ground_truth = query_data.get('gold', 'Unknown')
                    
                    # Make prediction
                    prediction, metadata = self.agent.predict(user_id, query_input)
                    
                    demo_count += 1
                    print(f"\n   Example {demo_count}:")
                    print(f"   User ID: {user_id}")
                    print(f"   Review: {query_input[:150]}{'...' if len(query_input) > 150 else ''}")
                    print(f"   Predicted Rating: {prediction} stars")
                    print(f"   Actual Rating: {ground_truth} stars")
                    print(f"   Correct: {'✓' if prediction == ground_truth else '✗'}")
                    
                    # Show personalization context
                    semantic_context = metadata.get('semantic_context', {})
                    user_interactions = semantic_context.get('user_interactions', [])
                    global_interactions = semantic_context.get('global_interactions', [])
                    category_prefs = semantic_context.get('category_preferences', {})
                    
                    if category_prefs:
                        top_prefs = sorted(category_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
                        prefs_str = ", ".join([f"{rating}★ ({pref:.2f})" for rating, pref in top_prefs])
                        print(f"   User's rating preferences: {prefs_str}")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise
    
    def run_evaluation(self):
        """Run evaluation on the product rating dataset."""
        print(f"\n📈 Running Product Rating Evaluation:")
        
        # Prepare evaluation data
        evaluation_queries = []
        evaluation_labels = {}
        
        for user_data in self.test_data:
            user_id = str(user_data.get('user_id', 'unknown'))
            queries = user_data.get('query', [])
            
            for query in queries:
                query_id = str(query.get('id', 'unknown'))
                query_input = query.get('input', '')
                ground_truth = str(query.get('gold', 'unknown'))
                
                eval_item = {
                    'user_id': user_id,
                    'id': query_id,
                    'input': query_input
                }
                evaluation_queries.append(eval_item)
                evaluation_labels[query_id] = ground_truth
        
        print(f"   Total rating queries to evaluate: {len(evaluation_queries)}")
        
        # Run predictions and collect prompts
        predictions = {}
        prompts_data = []
        correct = 0
        total = len(evaluation_queries)
        
        for i, eval_item in enumerate(evaluation_queries):
            user_id = eval_item['user_id']
            query_id = eval_item['id']
            query_input = eval_item['input']
            ground_truth = evaluation_labels[query_id]
            
            # Make prediction and get metadata including prompt
            prediction, metadata = self.agent.predict(user_id, query_input)
            predictions[query_id] = prediction
            
            # Collect prompt data for saving
            personalized_prompt = metadata.get('personalized_prompt', '')
            prompts_data.append({
                'query_id': query_id,
                'user_id': user_id,
                'prompt': personalized_prompt,
                'true_label': ground_truth,
                'predicted_label': prediction
            })
            
            if prediction == ground_truth:
                correct += 1
            
            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"   Progress: {i + 1}/{total} ({((i + 1)/total)*100:.1f}%)")
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Correct: {correct}/{total}")
        
        # Save prompts for external LLM evaluation
        self._save_prompts_for_external_llm(prompts_data)
        
        # Calculate per-rating performance
        rating_stats = {}
        for rating in self.config.categories:
            rating_stats[rating] = {'total': 0, 'correct': 0, 'predicted': 0}
        
        for query_id, ground_truth in evaluation_labels.items():
            if ground_truth in rating_stats:
                rating_stats[ground_truth]['total'] += 1
            
            prediction = predictions.get(query_id, 'unknown')
            if prediction == ground_truth and ground_truth in rating_stats:
                rating_stats[ground_truth]['correct'] += 1
            
            if prediction in rating_stats:
                rating_stats[prediction]['predicted'] += 1
        
        print(f"\n   📊 Per-Rating Performance:")
        for rating in sorted(rating_stats.keys()):
            stats = rating_stats[rating]
            precision = stats['correct'] / stats['predicted'] if stats['predicted'] > 0 else 0
            recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"     {rating} stars: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={stats['total']}")
        
        # Calculate Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) for rating prediction
        mae = self._calculate_mae(predictions, evaluation_labels)
        rmse = self._calculate_rmse(predictions, evaluation_labels)
        print(f"\n   📊 Rating-Specific Metrics:")
        print(f"     Mean Absolute Error (MAE): {mae:.3f}")
        print(f"     Root Mean Square Error (RMSE): {rmse:.3f}")
        
        # Save results
        evaluation = {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_examples': total,
            'mae': mae,
            'rmse': rmse,
            'rating_stats': rating_stats,
            'predictions': predictions,
            'labels': evaluation_labels,
            'prompts_data': prompts_data
        }
        
        saved_file = self._save_evaluation_results(evaluation)
        print(f"\n   💾 Results saved to: {saved_file}")
        
        return evaluation
    
    def _calculate_mae(self, predictions: Dict[str, str], labels: Dict[str, str]) -> float:
        """Calculate Mean Absolute Error for rating predictions."""
        total_error = 0
        count = 0
        
        for query_id, predicted_rating in predictions.items():
            actual_rating = labels.get(query_id)
            
            try:
                pred_val = float(predicted_rating)
                actual_val = float(actual_rating)
                total_error += abs(pred_val - actual_val)
                count += 1
            except (ValueError, TypeError):
                continue  # Skip invalid ratings
        
        return total_error / count if count > 0 else 0.0
    
    def _calculate_rmse(self, predictions: Dict[str, str], labels: Dict[str, str]) -> float:
        """Calculate Root Mean Square Error for rating predictions."""
        total_squared_error = 0
        count = 0
        
        for query_id, predicted_rating in predictions.items():
            actual_rating = labels.get(query_id)
            
            try:
                pred_val = float(predicted_rating)
                actual_val = float(actual_rating)
                squared_error = (pred_val - actual_val) ** 2
                total_squared_error += squared_error
                count += 1
            except (ValueError, TypeError):
                continue  # Skip invalid ratings
        
        mse = total_squared_error / count if count > 0 else 0.0
        return mse ** 0.5  # Square root of MSE
    
    def _save_evaluation_results(self, evaluation: Dict[str, Any], filename: str = None) -> str:
        """Save evaluation results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"product_rating_results_{self.model_name}_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'experiment_info': {
                'task': 'product_rating',
                'model': self.model_name,
                'persona_weight': self.persona_weight,
                'sample_size': self.sample_size,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'train_file': self.config.train_file,
                    'test_file': self.config.test_file,
                    'categories': self.config.categories
                }
            },
            'metrics': {
                'accuracy': evaluation.get('accuracy', 0),
                'correct_predictions': evaluation.get('correct_predictions', 0),
                'total_examples': evaluation.get('total_examples', 0),
                'mae': evaluation.get('mae', 0),
                'rmse': evaluation.get('rmse', 0),
                'rating_stats': evaluation.get('rating_stats', {})
            },
            'predictions': evaluation.get('predictions', {}),
            'labels': evaluation.get('labels', {}),
            'results': evaluation.get('prompts_data', [])  # Include prompts data for external LLM evaluation
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        return filepath
    
    def _save_prompts_for_external_llm(self, prompts_data: List[Dict[str, Any]]) -> str:
        """
        Save prompts in a format compatible with external LLM evaluation (e.g., AWS Bedrock Claude).
        
        Args:
            prompts_data: List of dictionaries containing prompt data
            
        Returns:
            Path to the saved prompts file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"product_rating_prompts_{self.model_name}_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        # Prepare prompts data in the format expected by external LLM evaluation scripts
        prompts_for_external = {
            'experiment_info': {
                'task': 'product_rating',
                'model': self.model_name,
                'persona_weight': self.persona_weight,
                'sample_size': self.sample_size,
                'timestamp': datetime.now().isoformat(),
                'rating_categories': self.config.categories,
                'description': 'Generated prompts for product rating prediction task'
            },
            'results': []
        }
        
        # Convert prompts data to the expected format
        for item in prompts_data:
            result_item = {
                'query_id': item['query_id'],
                'user_id': item['user_id'],
                'prompt': item['prompt'],
                'true_label': item['true_label'],
                'predicted_label': item['predicted_label']
            }
            prompts_for_external['results'].append(result_item)
        
        # Save prompts file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompts_for_external, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Prompts saved for external LLM evaluation: {filepath}")
        print(f"   📊 Total prompts saved: {len(prompts_data)}")
        
        return filepath
    
    def run_complete_demo(self):
        """Run the complete product rating demo pipeline."""
        print("="*60)
        print(f" PRODUCT RATING PERSONAAGENT DEMO")
        print("="*60)
        print(f"🤖 Task: Product Rating Prediction (1-5 stars)")
        print(f"🧠 Model: {self.model_name}")
        print(f"⚙️  Persona Weight: {self.persona_weight}")
        
        # Load and analyze data
        self.load_data()
        self.analyze_dataset()
        
        # Build memory
        self.build_memory(num_processes=getattr(self, 'num_processes', None))
        
        # Demonstrate predictions
        self.demonstrate_predictions()
        
        # Run evaluation
        evaluation = self.run_evaluation()
        
        print("\n" + "="*60)
        print(" PRODUCT RATING DEMO COMPLETE")
        print("="*60)
        print(f"🎉 Product rating prediction demo completed!")
        print(f"   Final accuracy: {evaluation['accuracy']:.4f}")
        print(f"   Mean Absolute Error: {evaluation['mae']:.3f}")
        
        return evaluation


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Product Rating PersonaAgent Demo')
    
    parser.add_argument('--dataset', '-d', 
                       choices=list_available_product_rating_datasets(),
                       default='product_rating',
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
    
    parser.add_argument('--processes', '-p',
                       type=int,
                       help='Number of processes to use for parallel memory building')
    
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
        print("Available product rating datasets:")
        for dataset in list_available_product_rating_datasets():
            config = get_product_rating_dataset_config(dataset)
            print(f"  {dataset}: {config.name}")
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
        runner = ProductRatingRunner(
            dataset_name=args.dataset,
            model_name=args.model,
            ollama_host=args.host,
            persona_weight=args.persona_weight,
            sample_size=args.sample_size,
            custom_config=custom_config
        )
        
        # Set num_processes if specified
        if args.processes:
            runner.num_processes = args.processes
        
        runner.run_complete_demo()
        
    except Exception as e:
        print(f"❌ Product rating demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
