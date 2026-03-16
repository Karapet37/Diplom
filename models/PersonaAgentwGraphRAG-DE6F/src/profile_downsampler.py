"""
Profile Downsampler Utility

This module provides functionality to downsample user profiles to a maximum number of entries per user.
"""

import json
import random
from typing import Dict, List, Any, Optional


def downsample_profiles(data: List[Dict[str, Any]], max_profiles_per_user: int = 5, 
                       sampling_strategy: str = 'random', seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Downsample user profiles to a maximum number of entries per user.
    
    Args:
        data: List of user dictionaries, each containing 'user_id' and 'profile' fields
        max_profiles_per_user: Maximum number of profile entries to keep per user (default: 5)
        sampling_strategy: Strategy for selecting profiles ('random', 'recent', 'highest_rated')
        seed: Random seed for reproducible results when using random sampling
        
    Returns:
        List of user dictionaries with downsampled profiles
        
    Raises:
        ValueError: If max_profiles_per_user is less than 1
        KeyError: If required fields are missing from the data
    """
    if max_profiles_per_user < 1:
        raise ValueError("max_profiles_per_user must be at least 1")
    
    if seed is not None:
        random.seed(seed)
    
    downsampled_data = []
    
    for user_data in data:
        # Validate required fields
        if 'user_id' not in user_data:
            raise KeyError("Missing 'user_id' field in user data")
        if 'profile' not in user_data:
            raise KeyError("Missing 'profile' field in user data")
        
        user_id = user_data['user_id']
        profiles = user_data['profile']
        
        # If user already has max_profiles_per_user or fewer profiles, keep as is
        if len(profiles) <= max_profiles_per_user:
            downsampled_data.append(user_data.copy())
            continue
        
        # Select profiles based on strategy
        if sampling_strategy == 'random':
            selected_profiles = random.sample(profiles, max_profiles_per_user)
        elif sampling_strategy == 'recent':
            # Sort by date (assuming date format is YYYY-MM-DD) and take most recent
            sorted_profiles = sorted(profiles, key=lambda x: x.get('date', ''), reverse=True)
            selected_profiles = sorted_profiles[:max_profiles_per_user]
        elif sampling_strategy == 'highest_rated':
            # Sort by score and take highest rated
            sorted_profiles = sorted(profiles, key=lambda x: float(x.get('score', 0)), reverse=True)
            selected_profiles = sorted_profiles[:max_profiles_per_user]
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        # Create new user data with downsampled profiles
        downsampled_user = user_data.copy()
        downsampled_user['profile'] = selected_profiles
        downsampled_data.append(downsampled_user)
    
    return downsampled_data


def downsample_json_file(input_file: str, output_file: str, max_profiles_per_user: int = 5,
                        sampling_strategy: str = 'random', seed: Optional[int] = None) -> None:
    """
    Downsample profiles in a JSON file and save to a new file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        max_profiles_per_user: Maximum number of profile entries to keep per user
        sampling_strategy: Strategy for selecting profiles ('random', 'recent', 'highest_rated')
        seed: Random seed for reproducible results
    """
    try:
        # Load data from input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Downsample the data
        downsampled_data = downsample_profiles(data, max_profiles_per_user, sampling_strategy, seed)
        
        # Save to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(downsampled_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully downsampled {len(data)} users to {output_file}")
        print(f"Max profiles per user: {max_profiles_per_user}")
        print(f"Sampling strategy: {sampling_strategy}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
    except Exception as e:
        print(f"Error: {e}")


def get_profile_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about profile distribution in the dataset.
    
    Args:
        data: List of user dictionaries
        
    Returns:
        Dictionary containing statistics about the profiles
    """
    if not data:
        return {"total_users": 0, "total_profiles": 0}
    
    profile_counts = [len(user.get('profile', [])) for user in data]
    total_profiles = sum(profile_counts)
    
    stats = {
        "total_users": len(data),
        "total_profiles": total_profiles,
        "avg_profiles_per_user": total_profiles / len(data) if data else 0,
        "min_profiles_per_user": min(profile_counts) if profile_counts else 0,
        "max_profiles_per_user": max(profile_counts) if profile_counts else 0,
        "users_with_more_than_5_profiles": sum(1 for count in profile_counts if count > 5)
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Downsample user profiles in JSON data")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("output_file", help="Output JSON file path")
    parser.add_argument("--max-profiles", type=int, default=5, 
                       help="Maximum profiles per user (default: 5)")
    parser.add_argument("--strategy", choices=['random', 'recent', 'highest_rated'], 
                       default='random', help="Sampling strategy (default: random)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")
    parser.add_argument("--stats", action='store_true', 
                       help="Show statistics before and after downsampling")
    
    args = parser.parse_args()
    
    if args.stats:
        # Load and show original statistics
        with open(args.input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        print("Original data statistics:")
        original_stats = get_profile_statistics(original_data)
        for key, value in original_stats.items():
            print(f"  {key}: {value}")
        print()
    
    # Perform downsampling
    downsample_json_file(args.input_file, args.output_file, 
                        args.max_profiles, args.strategy, args.seed)
    
    if args.stats:
        # Load and show downsampled statistics
        with open(args.output_file, 'r', encoding='utf-8') as f:
            downsampled_data = json.load(f)
        
        print("\nDownsampled data statistics:")
        downsampled_stats = get_profile_statistics(downsampled_data)
        for key, value in downsampled_stats.items():
            print(f"  {key}: {value}")
