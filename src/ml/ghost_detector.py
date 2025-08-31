"""
src/ml/ghost_detector.py
PRISM Ghost Defaulter Detection System - Complete Implementation
Behavioral DNA fingerprinting to identify duplicate identities and ghost defaulters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import pickle
import hashlib
from datetime import datetime
from collections import defaultdict, Counter
import math

class BehavioralDNAExtractor:
    """
    Extracts behavioral DNA fingerprints from user activity patterns
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Feature dimensions
        self.movement_dim = 20
        self.transaction_dim = 15
        self.temporal_dim = 12
        self.social_dim = 10
        self.total_dim = self.movement_dim + self.transaction_dim + self.temporal_dim + self.social_dim
    
    def extract_movement_signature(self, movement_data: List[Dict]) -> np.ndarray:
        """Extract behavioral signature from movement patterns"""
        
        if not movement_data:
            return np.zeros(self.movement_dim)
        
        features = []
        
        # Location cluster distribution
        locations = [m.get('location_cluster_id', 0) for m in movement_data]
        location_counter = Counter(locations)
        total_records = len(locations)
        
        # Top 5 location frequencies (normalized)
        sorted_locations = sorted(location_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        location_ratios = []
        for i in range(5):
            if i < len(sorted_locations):
                location_ratios.append(sorted_locations[i][1] / total_records)
            else:
                location_ratios.append(0.0)
        features.extend(location_ratios)
        
        # Stay duration patterns
        stay_durations = [m.get('stay_duration', 60) for m in movement_data]
        features.extend([
            np.percentile(stay_durations, 25),  # Q1
            np.median(stay_durations),          # Q2
            np.percentile(stay_durations, 75),  # Q3
            np.std(stay_durations)              # Variability
        ])
        
        # Temporal movement patterns (hourly activity)
        timestamps = []
        for m in movement_data:
            if isinstance(m.get('timestamp'), str):
                try:
                    dt = datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00'))
                    timestamps.append(dt)
                except:
                    continue
        
        if timestamps:
            hours = [ts.hour for ts in timestamps]
            hour_counter = Counter(hours)
            # Peak activity hours (top 3)
            top_hours = sorted(hour_counter.items(), key=lambda x: x[1], reverse=True)[:3]
            for i in range(3):
                if i < len(top_hours):
                    features.append(top_hours[i][0] / 24.0)  # Normalized
                else:
                    features.append(0.5)
        else:
            features.extend([0.5, 0.5, 0.5])
        
        # Movement transition patterns
        if len(locations) > 1:
            transitions = []
            for i in range(1, len(locations)):
                if locations[i] != locations[i-1]:
                    transitions.append(1)
                else:
                    transitions.append(0)
            
            features.extend([
                np.mean(transitions),  # Mobility ratio
                len(set(locations)) / len(locations),  # Location diversity
                max(location_counter.values()) / total_records  # Dominant location ratio
            ])
        else:
            features.extend([0, 0, 0])
        
        # Weekly patterns
        if timestamps:
            weekdays = [ts.weekday() for ts in timestamps]
            weekday_counter = Counter(weekdays)
            weekend_ratio = (weekday_counter.get(5, 0) + weekday_counter.get(6, 0)) / len(weekdays)
            features.append(weekend_ratio)
        else:
            features.append(0.3)  # Default weekend ratio
        
        # Distance/transition analysis
        if len(movement_data) > 1:
            # Simulate distance between consecutive locations
            distances = []
            for i in range(1, len(movement_data)):
                # Simple location difference as proxy for distance
                loc_diff = abs(movement_data[i].get('location_cluster_id', 0) - 
                             movement_data[i-1].get('location_cluster_id', 0))
                distances.append(loc_diff)
            
            if distances:
                features.extend([np.mean(distances), np.std(distances)])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        return np.array(features[:self.movement_dim])
    
    def extract_transaction_signature(self, transaction_data: List[Dict]) -> np.ndarray:
        """Extract behavioral signature from transaction patterns"""
        
        if not transaction_data:
            return np.zeros(self.transaction_dim)
        
        features = []
        
        # Amount distribution patterns
        amounts = [float(t.get('amount', 0)) for t in transaction_data]
        log_amounts = [np.log1p(amt) for amt in amounts]
        
        features.extend([
            np.percentile(log_amounts, 25),
            np.median(log_amounts),
            np.percentile(log_amounts, 75),
            np.std(log_amounts)
        ])
        
        # Category preferences (top 5)
        categories = [t.get('category', 'unknown') for t in transaction_data]
        category_counter = Counter(categories)
        total_transactions = len(categories)
        
        sorted_categories = sorted(category_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        category_ratios = []
        for i in range(5):
            if i < len(sorted_categories):
                category_ratios.append(sorted_categories[i][1] / total_transactions)
            else:
                category_ratios.append(0.0)
        features.extend(category_ratios)
        
        # Payment method signature
        payment_methods = [t.get('payment_method', 'unknown') for t in transaction_data]
        payment_counter = Counter(payment_methods)
        upi_ratio = payment_counter.get('upi', 0) / total_transactions
        card_ratio = payment_counter.get('card', 0) / total_transactions
        features.extend([upi_ratio, card_ratio])
        
        # Timing patterns
        hours = [t.get('time_of_day', 12) for t in transaction_data]
        hour_counter = Counter(hours)
        peak_hours = sorted(hour_counter.items(), key=lambda x: x[1], reverse=True)[:2]
        
        for i in range(2):
            if i < len(peak_hours):
                features.append(peak_hours[i][0] / 24.0)
            else:
                features.append(0.5)
        
        # Transaction frequency
        features.append(len(transaction_data) / 90.0)  # Transactions per day
        
        # Amount variability patterns
        if len(amounts) > 1:
            features.append(np.std(amounts) / (np.mean(amounts) + 1e-6))  # Coefficient of variation
        else:
            features.append(0.5)
        
        return np.array(features[:self.transaction_dim])
    
    def extract_temporal_signature(self, combined_data: List[Dict]) -> np.ndarray:
        """Extract temporal behavioral signature"""
        
        if not combined_data:
            return np.zeros(self.temporal_dim)
        
        features = []
        
        # Extract timestamps
        timestamps = []
        for record in combined_data:
            if isinstance(record.get('timestamp'), str):
                try:
                    ts = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                    timestamps.append(ts)
                except:
                    continue
        
        if not timestamps:
            return np.zeros(self.temporal_dim)
        
        # Hourly activity distribution
        hours = [ts.hour for ts in timestamps]
        hour_counter = Counter(hours)
        total_activity = len(timestamps)
        
        # Activity in different time blocks
        night_activity = sum(hour_counter.get(h, 0) for h in range(0, 6)) / total_activity
        morning_activity = sum(hour_counter.get(h, 0) for h in range(6, 12)) / total_activity
        afternoon_activity = sum(hour_counter.get(h, 0) for h in range(12, 18)) / total_activity
        evening_activity = sum(hour_counter.get(h, 0) for h in range(18, 24)) / total_activity
        
        features.extend([night_activity, morning_activity, afternoon_activity, evening_activity])
        
        # Peak activity hour and spread
        peak_hour = max(hour_counter, key=hour_counter.get) if hour_counter else 12
        features.append(peak_hour / 24.0)
        
        # Activity concentration (entropy-based)
        if hour_counter:
            hour_probs = [count/total_activity for count in hour_counter.values()]
            hour_entropy = -sum(p * np.log2(p) for p in hour_probs if p > 0)
            features.append(hour_entropy / np.log2(24))  # Normalized
        else:
            features.append(0.5)
        
        # Weekly patterns
        weekdays = [ts.weekday() for ts in timestamps]
        weekday_counter = Counter(weekdays)
        weekend_ratio = (weekday_counter.get(5, 0) + weekday_counter.get(6, 0)) / total_activity
        features.append(weekend_ratio)
        
        # Activity gaps analysis
        sorted_timestamps = sorted(timestamps)
        gaps = []
        for i in range(1, len(sorted_timestamps)):
            gap_hours = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds() / 3600
            gaps.append(gap_hours)
        
        if gaps:
            features.extend([
                np.median(gaps) / 24.0,  # Median gap in days
                np.std(gaps) / 24.0,     # Gap variability
                sum(1 for g in gaps if g > 72) / len(gaps),  # Long gap ratio
                sum(1 for g in gaps if g < 1) / len(gaps)    # Very short gap ratio
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features[:self.temporal_dim])
    
    def extract_social_signature(self, social_data: List[Dict]) -> np.ndarray:
        """Extract social behavioral signature"""
        
        if not social_data:
            return np.zeros(self.social_dim)
        
        features = []
        
        # Connection type distribution
        connection_types = [s.get('connection_type', 'unknown') for s in social_data]
        type_counter = Counter(connection_types)
        total_connections = len(connection_types)
        
        # Key connection type ratios
        family_ratio = type_counter.get('family', 0) / total_connections
        friend_ratio = type_counter.get('friend', 0) / total_connections
        transaction_ratio = type_counter.get('transaction', 0) / total_connections
        features.extend([family_ratio, friend_ratio, transaction_ratio])
        
        # Connection strength patterns
        strengths = [s.get('strength', 0.5) for s in social_data]
        features.extend([
            np.mean(strengths),
            np.std(strengths),
            sum(1 for s in strengths if s > 0.7) / len(strengths)  # Strong connection ratio
        ])
        
        # Interaction frequency patterns
        frequencies = [s.get('frequency', 1) for s in social_data]
        features.extend([
            np.log1p(np.mean(frequencies)),
            np.std(frequencies) / (np.mean(frequencies) + 1e-6),  # CV
            sum(1 for f in frequencies if f > 10) / len(frequencies)  # High freq ratio
        ])
        
        # Risk correlation signature
        risk_correlations = [s.get('risk_correlation', 0.5) for s in social_data]
        features.append(np.mean(risk_correlations))
        
        return np.array(features[:self.social_dim])
    
    def create_behavioral_fingerprint(self, user_data: Dict) -> Dict:
        """Create complete behavioral DNA fingerprint"""
        
        # Extract component signatures
        movement_sig = self.extract_movement_signature(user_data.get('movement_data', []))
        transaction_sig = self.extract_transaction_signature(user_data.get('transaction_data', []))
        
        combined_data = (user_data.get('movement_data', []) + 
                        user_data.get('transaction_data', []))
        temporal_sig = self.extract_temporal_signature(combined_data)
        social_sig = self.extract_social_signature(user_data.get('social_data', []))
        
        # Combine into composite fingerprint
        composite_fingerprint = np.concatenate([
            movement_sig, transaction_sig, temporal_sig, social_sig
        ])
        
        # Create hash
        fingerprint_string = ''.join([f"{x:.6f}" for x in composite_fingerprint])
        fingerprint_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()
        
        return {
            'user_id': user_data.get('user_id'),
            'fingerprint_hash': fingerprint_hash,
            'composite_vector': composite_fingerprint,
            'component_signatures': {
                'movement': movement_sig,
                'transaction': transaction_sig,
                'temporal': temporal_sig,
                'social': social_sig
            },
            'data_completeness': self._calculate_completeness(user_data)
        }
    
    def _calculate_completeness(self, user_data: Dict) -> float:
        """Calculate data completeness score"""
        scores = [
            min(1.0, len(user_data.get('movement_data', [])) / 100),
            min(1.0, len(user_data.get('transaction_data', [])) / 50),
            min(1.0, len(user_data.get('social_data', [])) / 10)
        ]
        return np.mean(scores)

class GhostDefaulterDetector:
    """
    Complete Ghost Defaulter Detection System
    """
    
    def __init__(self):
        self.dna_extractor = BehavioralDNAExtractor()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clustering_model = DBSCAN(eps=0.15, min_samples=2)
        
        # Detection thresholds
        self.ghost_threshold = 0.85
        self.suspicious_threshold = 0.75
        
        # Storage
        self.fingerprint_database = {}
        self.known_defaulters = set()
        self.similarity_matrix = None
        
        # Model state
        self.is_fitted = False
    
    def fit(self, training_data: List[Dict]) -> 'GhostDefaulterDetector':
        """Train the ghost detection system"""
        
        print(f"Training Ghost Defaulter Detector on {len(training_data)} users...")
        
        # Create fingerprints for all training users
        fingerprints = []
        fingerprint_vectors = []
        
        for user_data in training_data:
            fingerprint = self.dna_extractor.create_behavioral_fingerprint(user_data)
            user_id = fingerprint['user_id']
            
            self.fingerprint_database[user_id] = fingerprint
            fingerprints.append(fingerprint)
            fingerprint_vectors.append(fingerprint['composite_vector'])
            
            # Track known defaulters
            if user_data.get('user_profile', {}).get('default_status') == 'defaulted':
                self.known_defaulters.add(user_id)
        
        if fingerprint_vectors:
            fingerprint_matrix = np.array(fingerprint_vectors)
            
            # Fit anomaly detector
            self.anomaly_detector.fit(fingerprint_matrix)
            
            # Fit DNA extractor scaler
            self.dna_extractor.scaler.fit(fingerprint_matrix)
            self.dna_extractor.is_fitted = True
            
            # Pre-compute similarity matrix for efficiency
            self.similarity_matrix = cosine_similarity(fingerprint_matrix)
            
            # Detect initial clusters
            cluster_labels = self.clustering_model.fit_predict(fingerprint_matrix)
            
            # Analyze cluster quality
            unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"Detected {unique_clusters} potential ghost clusters")
            
            self.is_fitted = True
            
        return self
    
    def detect_ghost_matches(self, user_id: str, threshold: float = None) -> List[Dict]:
        """Detect potential ghost matches for a user"""
        
        if threshold is None:
            threshold = self.ghost_threshold
        
        if user_id not in self.fingerprint_database:
            return []
        
        target_fingerprint = self.fingerprint_database[user_id]
        target_vector = target_fingerprint['composite_vector'].reshape(1, -1)
        
        matches = []
        
        for other_user_id, other_fingerprint in self.fingerprint_database.items():
            if other_user_id == user_id:
                continue
            
            other_vector = other_fingerprint['composite_vector'].reshape(1, -1)
            similarity = cosine_similarity(target_vector, other_vector)[0][0]
            
            if similarity >= threshold:
                
                # Component-wise similarity analysis
                component_similarities = {}
                for component in ['movement', 'transaction', 'temporal', 'social']:
                    comp1 = target_fingerprint['component_signatures'][component].reshape(1, -1)
                    comp2 = other_fingerprint['component_signatures'][component].reshape(1, -1)
                    comp_sim = cosine_similarity(comp1, comp2)[0][0]
                    component_similarities[component] = round(comp_sim, 4)
                
                # Calculate match confidence
                confidence = self._calculate_match_confidence(
                    target_fingerprint, other_fingerprint, similarity
                )
                
                # Risk assessment
                risk_factors = self._assess_risk_factors(other_user_id, similarity)
                
                match = {
                    'suspected_duplicate_user': other_user_id,
                    'similarity_score': round(similarity, 4),
                    'match_confidence': confidence,
                    'component_similarities': component_similarities,
                    'risk_factors': risk_factors,
                    'investigation_priority': self._determine_priority(similarity, risk_factors)
                }
                
                matches.append(match)
        
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)
    
    def _calculate_match_confidence(self, fp1: Dict, fp2: Dict, similarity: float) -> str:
        """Calculate confidence level for a match"""
        
        completeness1 = fp1['data_completeness']
        completeness2 = fp2['data_completeness']
        avg_completeness = (completeness1 + completeness2) / 2
        
        if similarity >= 0.9 and avg_completeness >= 0.7:
            return 'HIGH'
        elif similarity >= 0.85 and avg_completeness >= 0.5:
            return 'MEDIUM'
        elif similarity >= 0.75:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _assess_risk_factors(self, user_id: str, similarity: float) -> List[str]:
        """Assess risk factors for potential ghost"""
        
        risk_factors = []
        
        if user_id in self.known_defaulters:
            risk_factors.append("KNOWN_DEFAULTER")
        
        if similarity >= 0.95:
            risk_factors.append("EXTREMELY_HIGH_SIMILARITY")
        
        if user_id in self.fingerprint_database:
            completeness = self.fingerprint_database[user_id]['data_completeness']
            if completeness < 0.3:
                risk_factors.append("LOW_DATA_QUALITY")
        
        # Check for multiple high similarities
        similar_count = 0
        for other_id, other_fp in self.fingerprint_database.items():
            if other_id != user_id:
                other_vector = other_fp['composite_vector'].reshape(1, -1)
                user_vector = self.fingerprint_database[user_id]['composite_vector'].reshape(1, -1)
                sim = cosine_similarity(user_vector, other_vector)[0][0]
                if sim >= self.suspicious_threshold:
                    similar_count += 1
        
        if similar_count >= 2:
            risk_factors.append("MULTIPLE_SIMILAR_PROFILES")
        
        return risk_factors
    
    def _determine_priority(self, similarity: float, risk_factors: List[str]) -> str:
        """Determine investigation priority"""
        
        if "KNOWN_DEFAULTER" in risk_factors or similarity >= 0.9:
            return "URGENT"
        elif similarity >= 0.85 or len(risk_factors) >= 2:
            return "HIGH"
        elif similarity >= 0.75:
            return "MEDIUM"
        else:
            return "LOW"
    
    def detect_anomalous_behavior(self, user_data: Dict) -> Dict:
        """Detect anomalous behavioral patterns"""
        
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before anomaly detection")
        
        fingerprint = self.dna_extractor.create_behavioral_fingerprint(user_data)
        fingerprint_vector = fingerprint['composite_vector'].reshape(1, -1)
        
        # Anomaly score
        anomaly_score = self.anomaly_detector.decision_function(fingerprint_vector)[0]
        is_anomaly = self.anomaly_detector.predict(fingerprint_vector)[0] == -1
        
        # Scaled anomaly score to 0-1 range
        normalized_score = max(0, min(1, (anomaly_score + 0.5) / 1.0))
        
        return {
            'user_id': user_data.get('user_id'),
            'anomaly_score': round(float(anomaly_score), 4),
            'normalized_anomaly_score': round(normalized_score, 4),
            'is_anomaly': bool(is_anomaly),
            'anomaly_level': 'HIGH' if normalized_score > 0.7 else 'MEDIUM' if normalized_score > 0.4 else 'LOW'
        }
    
    def investigate_ghost_network(self, suspected_user: str) -> Dict:
        """Investigate potential network of ghost identities"""
        
        if suspected_user not in self.fingerprint_database:
            return {'error': 'User not found in database'}
        
        # Find all similar users
        similar_users = self.detect_ghost_matches(suspected_user, threshold=0.6)
        
        # Network analysis
        network_risk = 'LOW'
        if any('KNOWN_DEFAULTER' in u.get('risk_factors', []) for u in similar_users):
            network_risk = 'HIGH'
        elif len(similar_users) >= 3:
            network_risk = 'MEDIUM'
        
        # Generate recommendations
        recommendations = self._generate_investigation_recommendations(similar_users)
        
        return {
            'investigated_user': suspected_user,
            'similar_users_found': len(similar_users),
            'network_risk_level': network_risk,
            'similar_users': similar_users[:10],  # Top 10
            'investigation_recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_investigation_recommendations(self, similar_users: List[Dict]) -> List[str]:
        """Generate investigation recommendations"""
        
        recommendations = []
        
        if not similar_users:
            recommendations.append("No similar profiles detected - standard monitoring")
            return recommendations
        
        urgent_cases = [u for u in similar_users if u['investigation_priority'] == 'URGENT']
        if urgent_cases:
            recommendations.append(f"URGENT: {len(urgent_cases)} high-priority cases require immediate investigation")
            recommendations.append("Freeze all credit applications for suspected network")
            recommendations.append("Manual identity verification required")
        
        high_similarity = [u for u in similar_users if u['similarity_score'] >= 0.9]
        if high_similarity:
            recommendations.append(f"Extremely high similarity detected in {len(high_similarity)} cases")
            recommendations.append("Cross-verify identity documents and biometric data")
        
        if len(similar_users) >= 3:
            recommendations.append("Potential identity fraud ring detected")
            recommendations.append("Investigate shared addresses, phone numbers, and device fingerprints")
        
        recommendations.append("Enhanced KYC verification for all flagged profiles")
        
        return recommendations
    
    def generate_detection_report(self) -> Dict:
        """Generate comprehensive ghost detection report"""
        
        total_users = len(self.fingerprint_database)
        total_defaulters = len(self.known_defaulters)
        
        # Find all potential matches
        all_matches = []
        processed_pairs = set()
        
        for user_id in self.fingerprint_database:
            matches = self.detect_ghost_matches(user_id, threshold=0.65)
            
            for match in matches:
                pair = tuple(sorted([user_id, match['suspected_duplicate_user']]))
                if pair not in processed_pairs:
                    all_matches.append({
                        'user1': user_id,
                        'user2': match['suspected_duplicate_user'],
                        'similarity': match['similarity_score'],
                        'priority': match['investigation_priority']
                    })
                    processed_pairs.add(pair)
        
        # Categorize matches
        urgent_matches = [m for m in all_matches if m['priority'] == 'URGENT']
        high_priority = [m for m in all_matches if m['priority'] == 'HIGH']
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'detection_statistics': {
                'total_users_analyzed': total_users,
                'known_defaulters': total_defaulters,
                'potential_ghost_matches': len(all_matches),
                'urgent_investigations': len(urgent_matches),
                'high_priority_investigations': len(high_priority),
                'detection_rate': len(all_matches) / max(total_users, 1)
            },
            'urgent_matches': urgent_matches[:10],
            'high_priority_matches': high_priority[:20],
            'system_recommendations': self._generate_system_recommendations(all_matches)
        }
    
    def _generate_system_recommendations(self, all_matches: List[Dict]) -> List[str]:
        """Generate system-level recommendations"""
        
        recommendations = []
        detection_rate = len(all_matches) / max(len(self.fingerprint_database), 1)
        
        if detection_rate > 0.1:
            recommendations.append("HIGH ALERT: Unusually high ghost detection rate")
            recommendations.append("Review system integrity and data quality")
        
        urgent_count = len([m for m in all_matches if m['priority'] == 'URGENT'])
        if urgent_count > 0:
            recommendations.append(f"Process {urgent_count} urgent cases immediately")
        
        recommendations.append("Regular model retraining recommended")
        recommendations.append("Enhanced identity verification protocols advised")
        
        return recommendations
    
    def save_model(self, filepath: str):
        """Save the trained ghost detector"""
        model_data = {
            'dna_extractor': self.dna_extractor,
            'anomaly_detector': self.anomaly_detector,
            'clustering_model': self.clustering_model,
            'ghost_threshold': self.ghost_threshold,
            'suspicious_threshold': self.suspicious_threshold,
            'fingerprint_database': self.fingerprint_database,
            'known_defaulters': self.known_defaulters,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Ghost Detector saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained ghost detector"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.dna_extractor = model_data['dna_extractor']
        self.anomaly_detector = model_data['anomaly_detector']
        self.clustering_model = model_data['clustering_model']
        self.ghost_threshold = model_data['ghost_threshold']
        self.suspicious_threshold = model_data['suspicious_threshold']
        self.fingerprint_database = model_data['fingerprint_database']
        self.known_defaulters = model_data['known_defaulters']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Ghost Detector loaded from {filepath}")
        return self

# Example usage
if __name__ == "__main__":
    detector = GhostDefaulterDetector()
    
    # Sample training data
    sample_data = [
        {
            'user_id': 'USR001',
            'user_profile': {'default_status': 'defaulted'},
            'movement_data': [{'location_cluster_id': 0, 'timestamp': '2024-01-01T08:00:00'}],
            'transaction_data': [{'amount': 1250, 'category': 'grocery'}],
            'social_data': [{'connection_type': 'family', 'strength': 0.9}]
        },
        {
            'user_id': 'USR002',
            'user_profile': {'default_status': 'active'},
            'movement_data': [{'location_cluster_id': 0, 'timestamp': '2024-01-01T08:15:00'}],
            'transaction_data': [{'amount': 1280, 'category': 'grocery'}],
            'social_data': [{'connection_type': 'family', 'strength': 0.85}]
        }
    ]
    
    # Train detector
    detector.fit(sample_data)
    
    # Test ghost detection
    matches = detector.detect_ghost_matches('USR001')
    print(f"Ghost matches for USR001: {len(matches)}")
    
    # Generate report
    report = detector.generate_detection_report()
    print(f"Detection report: {report['detection_statistics']['potential_ghost_matches']} potential matches")