"""
src/ml/entropy_engine.py
PRISM Life Entropy Calculation Engine - Complete Implementation
Calculates Shannon entropy across movement, transaction, social, and temporal patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import math
from typing import Dict, List, Tuple, Optional
import json
import pickle
from sklearn.preprocessing import StandardScaler

class LifeEntropyEngine:
    """
    Core entropy calculation engine for PRISM credit risk assessment.
    Measures predictability and stability across four life dimensions.
    """
    
    def __init__(self):
        self.weights = {
            'movement': 0.3,
            'transaction': 0.3, 
            'social': 0.2,
            'temporal': 0.2
        }
        
        # Fitted scalers and parameters
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Cache for performance
        self.entropy_cache = {}
    
    def calculate_shannon_entropy(self, values: List) -> float:
        """
        Calculate Shannon entropy: H(X) = -Σ p(xi) * log2(p(xi))
        """
        if not values or len(values) == 0:
            return 0.0
            
        # Count occurrences
        value_counts = Counter(values)
        total_count = len(values)
        
        # Calculate probabilities and entropy
        entropy = 0.0
        for count in value_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize based on unique values
        unique_values = len(value_counts)
        max_possible_entropy = math.log2(unique_values) if unique_values > 1 else 1.0
        
        return min(entropy / max_possible_entropy, 1.0) if max_possible_entropy > 0 else 0.0
    
    def extract_movement_features(self, movement_data: List[Dict]) -> Dict:
        """Extract comprehensive movement entropy features"""
        if not movement_data:
            return {
                'movement_entropy': 0.0,
                'location_diversity': 0.0,
                'temporal_regularity': 0.0,
                'mobility_pattern': 0.0,
                'stay_duration_entropy': 0.0
            }
        
        # Location entropy
        locations = [m.get('location_cluster_id', 0) for m in movement_data]
        location_entropy = self.calculate_shannon_entropy(locations)
        
        # Temporal patterns
        timestamps = []
        for m in movement_data:
            if isinstance(m.get('timestamp'), str):
                try:
                    ts = datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00'))
                    timestamps.append(ts)
                except:
                    continue
            elif isinstance(m.get('timestamp'), datetime):
                timestamps.append(m['timestamp'])
        
        if timestamps:
            hours = [ts.hour for ts in timestamps]
            weekdays = [ts.weekday() for ts in timestamps]
            temporal_entropy = (self.calculate_shannon_entropy(hours) + 
                              self.calculate_shannon_entropy(weekdays)) / 2
        else:
            temporal_entropy = 0.5
        
        # Stay duration patterns
        stay_durations = [m.get('stay_duration', 60) for m in movement_data]
        # Bin durations for entropy calculation
        duration_bins = ['short' if d < 30 else 'medium' if d < 180 else 'long' for d in stay_durations]
        duration_entropy = self.calculate_shannon_entropy(duration_bins)
        
        # Location diversity
        unique_locations = len(set(locations))
        location_diversity = min(1.0, unique_locations / 10.0)  # Normalize to 0-1
        
        # Mobility pattern (transition frequency)
        transitions = 0
        for i in range(1, len(locations)):
            if locations[i] != locations[i-1]:
                transitions += 1
        mobility_pattern = transitions / len(locations) if locations else 0
        
        return {
            'movement_entropy': location_entropy,
            'location_diversity': location_diversity,
            'temporal_regularity': 1.0 - temporal_entropy,  # Higher regularity = lower entropy
            'mobility_pattern': mobility_pattern,
            'stay_duration_entropy': duration_entropy
        }
    
    def extract_transaction_features(self, transaction_data: List[Dict]) -> Dict:
        """Extract comprehensive transaction entropy features"""
        if not transaction_data:
            return {
                'transaction_entropy': 0.0,
                'category_diversity': 0.0,
                'amount_regularity': 0.0,
                'temporal_consistency': 0.0,
                'payment_method_entropy': 0.0
            }
        
        # Category entropy
        categories = [t.get('category', 'unknown') for t in transaction_data]
        category_entropy = self.calculate_shannon_entropy(categories)
        
        # Amount patterns - bin amounts for entropy
        amounts = [float(t.get('amount', 0)) for t in transaction_data]
        if amounts:
            # Create amount bins
            amount_bins = []
            for amt in amounts:
                if amt < 100: amount_bins.append('micro')
                elif amt < 500: amount_bins.append('small')
                elif amt < 2000: amount_bins.append('medium')
                elif amt < 10000: amount_bins.append('large')
                else: amount_bins.append('xlarge')
            amount_entropy = self.calculate_shannon_entropy(amount_bins)
            amount_regularity = 1.0 - amount_entropy
        else:
            amount_regularity = 0.0
        
        # Payment method entropy
        payment_methods = [t.get('payment_method', 'unknown') for t in transaction_data]
        payment_entropy = self.calculate_shannon_entropy(payment_methods)
        
        # Temporal consistency
        hours = [t.get('time_of_day', 12) for t in transaction_data]
        temporal_entropy = self.calculate_shannon_entropy(hours)
        temporal_consistency = 1.0 - temporal_entropy
        
        # Category diversity
        unique_categories = len(set(categories))
        category_diversity = min(1.0, unique_categories / 15.0)  # Normalize
        
        return {
            'transaction_entropy': category_entropy,
            'category_diversity': category_diversity,
            'amount_regularity': amount_regularity,
            'temporal_consistency': temporal_consistency,
            'payment_method_entropy': payment_entropy
        }
    
    def extract_social_features(self, social_data: List[Dict]) -> Dict:
        """Extract social network entropy features"""
        if not social_data:
            return {
                'social_entropy': 0.0,
                'connection_diversity': 0.0,
                'strength_consistency': 0.0,
                'risk_correlation_entropy': 0.0
            }
        
        # Connection type entropy
        connection_types = [s.get('connection_type', 'unknown') for s in social_data]
        type_entropy = self.calculate_shannon_entropy(connection_types)
        
        # Connection strength patterns
        strengths = [s.get('strength', 0.5) for s in social_data]
        strength_bins = ['weak' if s < 0.3 else 'medium' if s < 0.7 else 'strong' for s in strengths]
        strength_entropy = self.calculate_shannon_entropy(strength_bins)
        strength_consistency = 1.0 - strength_entropy
        
        # Risk correlation entropy
        risk_corrs = [s.get('risk_correlation', 0.5) for s in social_data]
        risk_bins = ['low' if r < 0.3 else 'medium' if r < 0.7 else 'high' for r in risk_corrs]
        risk_entropy = self.calculate_shannon_entropy(risk_bins)
        
        # Connection diversity
        unique_types = len(set(connection_types))
        connection_diversity = min(1.0, unique_types / 4.0)  # family, friend, transaction, contact
        
        return {
            'social_entropy': type_entropy,
            'connection_diversity': connection_diversity,
            'strength_consistency': strength_consistency,
            'risk_correlation_entropy': risk_entropy
        }
    
    def extract_temporal_features(self, combined_data: List[Dict]) -> Dict:
        """Extract temporal entropy features from all activity data"""
        if not combined_data:
            return {
                'temporal_entropy': 0.0,
                'hourly_regularity': 0.0,
                'weekly_consistency': 0.0,
                'activity_concentration': 0.0
            }
        
        timestamps = []
        for record in combined_data:
            if isinstance(record.get('timestamp'), str):
                try:
                    ts = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                    timestamps.append(ts)
                except:
                    continue
            elif isinstance(record.get('timestamp'), datetime):
                timestamps.append(record['timestamp'])
        
        if not timestamps:
            return {
                'temporal_entropy': 0.5,
                'hourly_regularity': 0.5,
                'weekly_consistency': 0.5,
                'activity_concentration': 0.5
            }
        
        # Hour distribution
        hours = [ts.hour for ts in timestamps]
        hourly_entropy = self.calculate_shannon_entropy(hours)
        hourly_regularity = 1.0 - hourly_entropy
        
        # Weekly patterns
        weekdays = [ts.weekday() for ts in timestamps]
        weekly_entropy = self.calculate_shannon_entropy(weekdays)
        weekly_consistency = 1.0 - weekly_entropy
        
        # Activity concentration (how concentrated activity is in peak hours)
        hour_counts = Counter(hours)
        total_activity = len(hours)
        top_3_hours = sum(sorted(hour_counts.values(), reverse=True)[:3])
        activity_concentration = top_3_hours / total_activity if total_activity > 0 else 0
        
        # Overall temporal entropy
        temporal_entropy = (hourly_entropy + weekly_entropy) / 2
        
        return {
            'temporal_entropy': temporal_entropy,
            'hourly_regularity': hourly_regularity,
            'weekly_consistency': weekly_consistency,
            'activity_concentration': activity_concentration
        }
    
    def fit(self, training_data: List[Dict]) -> 'LifeEntropyEngine':
        """Fit the entropy engine on training data"""
        print("Fitting Life Entropy Engine...")
        
        all_features = []
        for user_data in training_data:
            features = self.extract_all_features(user_data)
            feature_vector = self._features_to_vector(features)
            all_features.append(feature_vector)
        
        if all_features:
            feature_matrix = np.array(all_features)
            self.scaler.fit(feature_matrix)
            self.is_fitted = True
            print(f"Entropy engine fitted on {len(all_features)} samples")
        
        return self
    
    def extract_all_features(self, user_data: Dict) -> Dict:
        """Extract all entropy features for a user"""
        movement_features = self.extract_movement_features(user_data.get('movement_data', []))
        transaction_features = self.extract_transaction_features(user_data.get('transaction_data', []))
        social_features = self.extract_social_features(user_data.get('social_data', []))
        
        # Combine movement and transaction data for temporal analysis
        combined_data = (user_data.get('movement_data', []) + 
                        user_data.get('transaction_data', []))
        temporal_features = self.extract_temporal_features(combined_data)
        
        return {
            **movement_features,
            **transaction_features, 
            **social_features,
            **temporal_features
        }
    
    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dictionary to vector"""
        feature_order = [
            'movement_entropy', 'location_diversity', 'temporal_regularity', 'mobility_pattern', 'stay_duration_entropy',
            'transaction_entropy', 'category_diversity', 'amount_regularity', 'temporal_consistency', 'payment_method_entropy',
            'social_entropy', 'connection_diversity', 'strength_consistency', 'risk_correlation_entropy',
            'hourly_regularity', 'weekly_consistency', 'activity_concentration'
        ]
        
        return np.array([features.get(f, 0.0) for f in feature_order])
    
    def calculate_life_stability_index(self, user_data: Dict) -> Dict:
        """Calculate comprehensive Life Stability Index for a user"""
        
        # Extract features
        features = self.extract_all_features(user_data)
        
        # Calculate component scores
        movement_score = (
            0.4 * features['movement_entropy'] +
            0.2 * (1.0 - features['location_diversity']) +
            0.2 * features['temporal_regularity'] +
            0.2 * (1.0 - features['mobility_pattern'])
        )
        
        transaction_score = (
            0.3 * features['transaction_entropy'] +
            0.2 * (1.0 - features['category_diversity']) +
            0.3 * features['amount_regularity'] +
            0.2 * features['temporal_consistency']
        )
        
        social_score = (
            0.4 * features['social_entropy'] +
            0.2 * (1.0 - features['connection_diversity']) +
            0.2 * features['strength_consistency'] +
            0.2 * (1.0 - features['risk_correlation_entropy'])
        )
        
        temporal_score = (
            0.4 * features['hourly_regularity'] +
            0.3 * features['weekly_consistency'] +
            0.3 * features['activity_concentration']
        )
        
        # Composite Life Stability Index
        composite_lsi = (
            self.weights['movement'] * movement_score +
            self.weights['transaction'] * transaction_score +
            self.weights['social'] * social_score +
            self.weights['temporal'] * temporal_score
        )
        
        # Interpret stability (higher LSI = more stable = lower risk)
        if composite_lsi >= 0.7:
            stability_rating = 'HIGH'
            risk_implication = 'LOW'
        elif composite_lsi >= 0.5:
            stability_rating = 'MODERATE'
            risk_implication = 'MEDIUM'
        elif composite_lsi >= 0.3:
            stability_rating = 'LOW'
            risk_implication = 'HIGH'
        else:
            stability_rating = 'VERY_LOW'
            risk_implication = 'VERY_HIGH'
        
        return {
            'user_id': user_data.get('user_id'),
            'component_scores': {
                'movement': round(movement_score, 4),
                'transaction': round(transaction_score, 4),
                'social': round(social_score, 4),
                'temporal': round(temporal_score, 4)
            },
            'composite_lsi': round(composite_lsi, 4),
            'stability_rating': stability_rating,
            'risk_implication': risk_implication,
            'detailed_features': features,
            'calculation_timestamp': datetime.now().isoformat()
        }
    
    def predict_entropy_scores(self, user_data: Dict) -> Dict:
        """Predict individual entropy scores for API use"""
        result = self.calculate_life_stability_index(user_data)
        
        return {
            'movement_entropy': 1.0 - result['component_scores']['movement'],  # Convert stability to entropy
            'transaction_entropy': 1.0 - result['component_scores']['transaction'],
            'social_entropy': 1.0 - result['component_scores']['social'], 
            'temporal_entropy': 1.0 - result['component_scores']['temporal'],
            'composite_entropy': 1.0 - result['composite_lsi'],
            'stability_rating': result['stability_rating']
        }
    
    def save_model(self, filepath: str):
        """Save the fitted entropy engine"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'weights': self.weights,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Entropy engine saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted entropy engine"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.weights = data['weights']
            self.is_fitted = data['is_fitted']
        print(f"Entropy engine loaded from {filepath}")
        return self

# Example usage
if __name__ == "__main__":
    # Test with sample data
    entropy_engine = LifeEntropyEngine()
    
    sample_user_data = {
        'user_id': 'USR00001',
        'movement_data': [
            {'location_cluster_id': 0, 'timestamp': '2024-01-01T08:00:00', 'stay_duration': 120},
            {'location_cluster_id': 1, 'timestamp': '2024-01-01T10:00:00', 'stay_duration': 480},
        ],
        'transaction_data': [
            {'category': 'grocery', 'amount': 1250.50, 'payment_method': 'upi', 'time_of_day': 11}
        ],
        'social_data': [
            {'connection_type': 'family', 'strength': 0.9, 'risk_correlation': 0.2}
        ]
    }
    
    result = entropy_engine.calculate_life_stability_index(sample_user_data)
    print(f"Life Stability Index: {result['composite_lsi']}")
    print(f"Stability Rating: {result['stability_rating']}")