import csv
import math
import os
from collections import defaultdict

class SimpleMarketingRecommender:
    def __init__(self, data_path='data/marketing_strategies.csv'):
        self.data_path = data_path
        self.strategies = []
        self.load_data()
    
    def load_data(self):
        """Load marketing strategies data from CSV"""
        if not os.path.exists(self.data_path):
            print(f"Error: Data file not found at {self.data_path}")
            return
            
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.strategies = list(reader)
                
            # Convert numeric strings to integers
            numeric_fields = [
                'strategy_id', 'budget_required', 'technical_expertise', 
                'time_investment', 'conversion_rate', 'brand_awareness', 
                'lead_generation', 'customer_retention', 'target_audience_size'
            ]
            
            for strategy in self.strategies:
                for field in numeric_fields:
                    strategy[field] = int(strategy[field])
                    
            print(f"Loaded {len(self.strategies)} marketing strategies")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
            
        # Return cosine similarity
        return dot_product / (magnitude1 * magnitude2)
    
    def get_recommendations(self, user_preferences, top_n=3):
        """
        Get personalized recommendations based on user preferences
        
        Args:
            user_preferences: dict with the following keys:
                - budget_level: 1-5 (1=very low, 5=very high)
                - technical_skill: 1-5 (1=beginner, 5=expert)
                - time_available: 1-5 (1=very little, 5=abundant)
                - goal: one of ['conversion', 'awareness', 'leads', 'retention']
                - industry: string with the industry name
                - audience_size: 1-5 (1=very small, 5=very large)
            top_n: number of recommendations to return
            
        Returns:
            List of recommended strategies
        """
        # Create user profile vector
        user_profile = [0] * 8
        
        # Map budget to budget_required (inverse relationship)
        user_profile[0] = 6 - user_preferences['budget_level']  # Inverse: higher budget = lower concern
        
        # Map technical skill to technical_expertise (inverse relationship)
        user_profile[1] = 6 - user_preferences['technical_skill']  # Inverse: higher skill = lower concern
        
        # Map time available to time_investment (inverse relationship)
        user_profile[2] = 6 - user_preferences['time_available']  # Inverse: more time = lower concern
        
        # Map goal to specific features
        goal_mapping = {
            'conversion': 3,  # index of conversion_rate
            'awareness': 4,   # index of brand_awareness
            'leads': 5,       # index of lead_generation
            'retention': 6    # index of customer_retention
        }
        
        # Set default values for goals
        for i in range(3, 7):
            user_profile[i] = 3  # Set default value
            
        # Emphasize the specific goal
        if user_preferences['goal'] in goal_mapping:
            goal_index = goal_mapping[user_preferences['goal']]
            user_profile[goal_index] = 5  # Boost the specific goal
        
        # Map audience size
        user_profile[7] = user_preferences['audience_size']
        
        # Calculate similarity with all strategies
        strategy_similarities = []
        
        for strategy in self.strategies:
            # Extract strategy features
            strategy_features = [
                strategy['budget_required'],
                strategy['technical_expertise'],
                strategy['time_investment'],
                strategy['conversion_rate'],
                strategy['brand_awareness'],
                strategy['lead_generation'],
                strategy['customer_retention'],
                strategy['target_audience_size']
            ]
            
            # Calculate similarity
            similarity = self.cosine_similarity(user_profile, strategy_features)
            
            # Check if industry matches
            industry_match = False
            if user_preferences.get('industry'):
                industries = strategy['best_for_industry'].replace('"', '').split(',')
                industry_match = user_preferences['industry'] in industries or 'All' in industries
            else:
                industry_match = True  # No industry filter
                
            # Add to results if industry matches
            if industry_match:
                strategy_similarities.append((strategy, similarity))
        
        # Sort by similarity score (descending)
        strategy_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        return strategy_similarities[:top_n]
    
    def get_strategy_details(self, strategy_id):
        """Get detailed information about a specific strategy"""
        for strategy in self.strategies:
            if strategy['strategy_id'] == strategy_id:
                return strategy
        return None

# Example usage
if __name__ == "__main__":
    recommender = SimpleMarketingRecommender()
    
    user_prefs = {
        'budget_level': 3,        # Medium budget
        'technical_skill': 2,     # Low technical skills
        'time_available': 4,      # Good amount of time
        'goal': 'awareness',      # Primary goal is brand awareness
        'industry': 'Fashion',    # Fashion industry
        'audience_size': 4        # Large audience
    }
    
    recommendations = recommender.get_recommendations(user_prefs, top_n=3)
    
    print("\nTop Recommendations:")
    for i, (strategy, similarity) in enumerate(recommendations, 1):
        print(f"{i}. {strategy['strategy_name']} (Similarity: {similarity:.2f})")
        print(f"   Best for: {strategy['best_for_industry']}")
        print(f"   Budget required: {strategy['budget_required']}/5")
        print(f"   Technical expertise: {strategy['technical_expertise']}/5")
        print(f"   Time investment: {strategy['time_investment']}/5")
        print(f"   Brand awareness impact: {strategy['brand_awareness']}/5")
        print()