"""
Parallel Memory Builder for PersonaAgent GraphRAG

This module provides parallel processing capabilities for building GraphRAG memory
to speed up the memory construction process using multiple CPU cores.
"""

import json
import multiprocessing as mp
from typing import Dict, List, Any, Tuple
from functools import partial
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
import hashlib
import re
from collections import defaultdict

# Import the original classes
from persona_agent import Interaction, GraphRAGMemory


@dataclass
class UserBatch:
    """Represents a batch of users to process in parallel."""
    user_data: List[Dict[str, Any]]
    batch_id: int


def process_user_batch(batch: UserBatch, categories: List[str]) -> Tuple[int, Dict[str, Any]]:
    """
    Process a batch of users and return their memory components.
    
    Args:
        batch: UserBatch containing user data to process
        categories: List of available categories
        
    Returns:
        Tuple of (batch_id, processed_data)
    """
    # Create a temporary memory instance for this batch
    temp_memory = GraphRAGMemory()
    
    processed_users = {}
    
    for user_data in batch.user_data:
        user_id = str(user_data.get('id', user_data.get('user_id', 'unknown')))
        profile = user_data.get('profile', [])
        
        # Convert profile to standard format and add to memory
        interactions = []
        for i, interaction_data in enumerate(profile):
            # Handle different data formats
            interaction_id = interaction_data.get('id', f"{user_id}_{i}")
            interaction_text = interaction_data.get('text', '')
            interaction_title = interaction_data.get('title', '')
            interaction_category = interaction_data.get('category', interaction_data.get('score', 'unknown'))
            
            # Ensure we have valid text content
            if not interaction_text and not interaction_title:
                # Skip empty interactions
                continue
                
            interaction = Interaction(
                id=str(interaction_id),
                text=str(interaction_text),
                title=str(interaction_title),
                category=str(interaction_category)
            )
            interactions.append(interaction)
            temp_memory.add_interaction(user_id, interaction)
        
        processed_users[user_id] = {
            'interactions': interactions,
            'interaction_count': len(interactions)
        }
    
    # Extract the components we need to merge back
    batch_data = {
        'nodes': dict(temp_memory.nodes),
        'graph_edges': list(temp_memory.graph.edges(data=True)),
        'user_interactions': dict(temp_memory.user_interactions),
        'interaction_texts': temp_memory.interaction_texts.copy(),
        'interaction_ids': temp_memory.interaction_ids.copy(),
        'processed_users': processed_users
    }
    
    return batch.batch_id, batch_data


class ParallelMemoryBuilder:
    """
    Parallel memory builder for GraphRAG system.
    """
    
    def __init__(self, num_processes: int = None):
        """
        Initialize the parallel memory builder.
        
        Args:
            num_processes: Number of processes to use (default: CPU count)
        """
        self.num_processes = num_processes or mp.cpu_count()
        self.logger = logging.getLogger(__name__)
        
    def build_memory_parallel(self, 
                            train_data: List[Dict[str, Any]], 
                            test_data: List[Dict[str, Any]] = None,
                            categories: List[str] = None) -> GraphRAGMemory:
        """
        Build GraphRAG memory using parallel processing.
        
        Args:
            train_data: Training data with user interactions
            test_data: Optional test data with user interactions
            categories: List of available categories
            
        Returns:
            Populated GraphRAGMemory instance
        """
        categories = categories or ["1", "2", "3", "4", "5"]  # Default for product rating
        
        # Combine train and test data
        all_data = train_data.copy()
        if test_data:
            all_data.extend(test_data)
        
        self.logger.info(f"Building memory for {len(all_data)} users using {self.num_processes} processes")
        
        # Split data into batches for parallel processing
        batch_size = max(1, len(all_data) // self.num_processes)
        batches = []
        
        for i in range(0, len(all_data), batch_size):
            batch_data = all_data[i:i + batch_size]
            batch = UserBatch(user_data=batch_data, batch_id=len(batches))
            batches.append(batch)
        
        self.logger.info(f"Created {len(batches)} batches with ~{batch_size} users each")
        
        # Process batches in parallel
        with mp.Pool(processes=self.num_processes) as pool:
            process_func = partial(process_user_batch, categories=categories)
            results = pool.map(process_func, batches)
        
        # Merge results into a single memory instance
        memory = self._merge_batch_results(results)
        
        self.logger.info(f"Memory building complete: {len(memory.user_interactions)} users, "
                        f"{len(memory.nodes)} nodes, {len(memory.interaction_texts)} interactions")
        
        return memory
    
    def _merge_batch_results(self, results: List[Tuple[int, Dict[str, Any]]]) -> GraphRAGMemory:
        """
        Merge batch processing results into a single GraphRAGMemory instance.
        
        Args:
            results: List of (batch_id, batch_data) tuples
            
        Returns:
            Merged GraphRAGMemory instance
        """
        memory = GraphRAGMemory()
        
        # Sort results by batch_id to maintain order
        results.sort(key=lambda x: x[0])
        
        total_interactions = 0
        
        for batch_id, batch_data in results:
            # Merge nodes
            for node_id, node in batch_data['nodes'].items():
                memory.nodes[node_id] = node
                memory.graph.add_node(node_id, **node.properties)
            
            # Merge graph edges
            for edge in batch_data['graph_edges']:
                if len(edge) == 3:  # (node1, node2, data)
                    memory.graph.add_edge(edge[0], edge[1], **edge[2])
                else:  # (node1, node2)
                    memory.graph.add_edge(edge[0], edge[1])
            
            # Merge user interactions
            for user_id, interaction_ids in batch_data['user_interactions'].items():
                memory.user_interactions[user_id].extend(interaction_ids)
            
            # Merge interaction texts and IDs
            memory.interaction_texts.extend(batch_data['interaction_texts'])
            memory.interaction_ids.extend(batch_data['interaction_ids'])
            
            total_interactions += len(batch_data['interaction_texts'])
            
            self.logger.info(f"Merged batch {batch_id}: {len(batch_data['processed_users'])} users, "
                           f"{len(batch_data['interaction_texts'])} interactions")
        
        # Refit TF-IDF vectorizer with all texts
        if memory.interaction_texts:
            try:
                memory.tfidf_vectorizer.fit(memory.interaction_texts)
                self.logger.info("TF-IDF vectorizer fitted successfully")
            except ValueError as e:
                self.logger.warning(f"TF-IDF fitting failed: {e}")
        
        return memory


def build_memory_for_demo(train_file: str, 
                         test_file: str = None, 
                         categories: List[str] = None,
                         num_processes: int = None) -> GraphRAGMemory:
    """
    Convenience function to build memory from data files using parallel processing.
    
    Args:
        train_file: Path to training data JSON file
        test_file: Optional path to test data JSON file
        categories: List of available categories
        num_processes: Number of processes to use
        
    Returns:
        Populated GraphRAGMemory instance
    """
    # Load data
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    test_data = None
    if test_file:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    
    # Build memory in parallel
    builder = ParallelMemoryBuilder(num_processes=num_processes)
    memory = builder.build_memory_parallel(train_data, test_data, categories)
    
    return memory


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build GraphRAG memory in parallel')
    parser.add_argument('train_file', help='Training data JSON file')
    parser.add_argument('--test-file', help='Test data JSON file (optional)')
    parser.add_argument('--processes', '-p', type=int, help='Number of processes to use')
    parser.add_argument('--categories', nargs='+', default=["1", "2", "3", "4", "5"],
                       help='List of categories')
    parser.add_argument('--output', '-o', help='Output file to save memory (optional)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Build memory
    memory = build_memory_for_demo(
        train_file=args.train_file,
        test_file=args.test_file,
        categories=args.categories,
        num_processes=args.processes
    )
    
    print(f"\n=== Memory Building Complete ===")
    print(f"Total users: {len(memory.user_interactions)}")
    print(f"Total nodes: {len(memory.nodes)}")
    print(f"Total interactions: {len(memory.interaction_texts)}")
    print(f"Graph edges: {memory.graph.number_of_edges()}")
    
    if args.output:
        # Save memory state (simplified - you might want to implement proper serialization)
        memory_state = {
            'user_count': len(memory.user_interactions),
            'node_count': len(memory.nodes),
            'interaction_count': len(memory.interaction_texts),
            'edge_count': memory.graph.number_of_edges()
        }
        
        with open(args.output, 'w') as f:
            json.dump(memory_state, f, indent=2)
        
        print(f"Memory statistics saved to: {args.output}")
