"""
src/ml/ensemble_model.py
PRISM Ensemble Model - FIXED Complete Implementation
Combines all ML components (Entropy, Social Risk, Ghost Detection) for final risk assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our ML components
from .entropy_engine import LifeEntropyEngine
from .social_risk_gnn import SocialRiskAnalyzer  
from .ghost_detector import GhostDefaulterDetector

class PRISMEnsembleModel:
    """
    Master ensemble model that combines all PRISM components for final risk assessment
    """
    
    def __init__(self):
        # Component models
        self.entropy_engine = LifeEntropyEngine()
        self.social_analyzer = SocialRiskAnalyzer()
        self.ghost_detector = GhostDefaulterDetector()
        
        # Ensemble models
        self.default_classifier = GradientBoostingClassifier(
            n_estimators=150, 
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        self.risk_score_regressor = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=6, 
            learning_rate=0.1,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        # Feature scaling
        self.ensemble_scaler = StandardScaler()
        
        # Model weights and thresholds
        self.component_weights = {
            'entropy': 0.35,
            'social': 0.30,
            'ghost': 0.15,
            'meta_features': 0.20
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.is_fitted = False
        
        # Feature importance tracking
        self.feature_importance = {}
    
    def extract_ensemble_features(self, user_data: Dict) -> Dict:
        """Extract comprehensive features from all model components - FIXED VERSION"""
        
        features = {}
        
        # 1. Entropy Engine Features (safe - uses direct calculation)
        try:
            entropy_result = self.entropy_engine.calculate_life_stability_index(user_data)
            features.update({
                'movement_entropy': 1.0 - entropy_result['component_scores']['movement'],
                'transaction_entropy': 1.0 - entropy_result['component_scores']['transaction'],
                'social_entropy': 1.0 - entropy_result['component_scores']['social'],
                'temporal_entropy': 1.0 - entropy_result['component_scores']['temporal'],
                'composite_lsi': entropy_result['composite_lsi'],
                'stability_rating_numeric': {
                    'HIGH': 0.8, 'MODERATE': 0.5, 'LOW': 0.2, 'VERY_LOW': 0.1
                }.get(entropy_result['stability_rating'], 0.5)
            })
        except Exception as e:
            print(f"Entropy engine error: {e}")
            features.update({
                'movement_entropy': 0.5, 'transaction_entropy': 0.5,
                'social_entropy': 0.5, 'temporal_entropy': 0.5,
                'composite_lsi': 0.5, 'stability_rating_numeric': 0.5
            })
        
        # 2. Social Features (simplified - no prediction calls during training)
        social_data = user_data.get('social_data', [])
        if social_data:
            strengths = [s.get('strength', 0.5) for s in social_data]
            risk_corrs = [s.get('risk_correlation', 0.5) for s in social_data]
            
            features.update({
                'social_risk_score': min(1.0, len(social_data) / 20.0),
                'network_connections': len(social_data),
                'avg_connection_strength': np.mean(strengths),
                'max_connection_strength': np.max(strengths),
                'avg_risk_correlation': np.mean(risk_corrs),
                'high_risk_connections': sum(1 for r in risk_corrs if r > 0.6),
                'cascade_risk_estimate': np.mean(risk_corrs)
            })
        else:
            features.update({
                'social_risk_score': 0.1, 'network_connections': 0,
                'avg_connection_strength': 0.0, 'max_connection_strength': 0.0,
                'avg_risk_correlation': 0.0, 'high_risk_connections': 0,
                'cascade_risk_estimate': 0.0
            })
        
        # 3. Ghost Features (simplified - no complex fingerprint matching during training)
        movement_count = len(user_data.get('movement_data', []))
        transaction_count = len(user_data.get('transaction_data', []))
        social_count = len(user_data.get('social_data', []))
        
        # Simple behavioral diversity metrics
        data_completeness = self._calculate_data_completeness(user_data)
        behavioral_diversity = min(1.0, transaction_count / 50.0)
        
        # Simple anomaly indicators
        anomaly_score = 0.0
        if movement_count < 50:
            anomaly_score += 0.3
        if transaction_count < 10:
            anomaly_score += 0.3
        if social_count == 0:
            anomaly_score += 0.2
        
        features.update({
            'data_completeness': data_completeness,
            'behavioral_diversity': behavioral_diversity,
            'anomaly_score': min(1.0, anomaly_score),
            'ghost_risk_indicator': 1.0 if data_completeness < 0.3 else 0.0
        })
        
        # 4. Meta Features (user profile and data quality)
        user_profile = user_data.get('user_profile', {})
        features.update({
            'age': user_profile.get('age', 30) / 100.0,  # Normalized
            'income_bracket_numeric': {
                'low': 0.2, 'medium': 0.5, 'high': 0.8
            }.get(user_profile.get('income_bracket', 'medium'), 0.5),
            'movement_record_count': movement_count,
            'transaction_record_count': transaction_count,
            'social_connection_count': social_count,
            'account_age_days': self._calculate_account_age(user_profile),
            'existing_risk_score': user_profile.get('credit_risk_score', 600) / 1000.0  # Normalized
        })
        
        # 5. Interaction features
        features.update({
            'data_social_interaction': data_completeness * features['social_risk_score'],
            'entropy_social_interaction': features['composite_lsi'] * features['social_risk_score'],
            'volume_diversity_ratio': (movement_count + transaction_count) / max(1, social_count),
            'comprehensive_score': data_completeness * features['composite_lsi'] * (1.0 - features['social_risk_score'])
        })
        
        return features
    
    def _calculate_data_completeness(self, user_data: Dict) -> float:
        """Calculate overall data completeness score"""
        scores = [
            min(1.0, len(user_data.get('movement_data', [])) / 100),
            min(1.0, len(user_data.get('transaction_data', [])) / 50), 
            min(1.0, len(user_data.get('social_data', [])) / 10)
        ]
        return np.mean(scores)
    
    def _calculate_account_age(self, user_profile: Dict) -> float:
        """Calculate account age in normalized days"""
        try:
            if 'created_at' in user_profile:
                if isinstance(user_profile['created_at'], str):
                    created_date = datetime.fromisoformat(user_profile['created_at'].replace('Z', '+00:00'))
                else:
                    created_date = user_profile['created_at']
                
                age_days = (datetime.now() - created_date).days
                return min(1.0, age_days / 365.0)  # Normalize to max 1 year
            else:
                return 0.5  # Default middle value
        except:
            return 0.5
    
    def prepare_feature_matrix(self, training_data: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from training data"""
        
        print("Extracting ensemble features from training data...")
        
        all_features = []
        feature_names = None
        
        for i, user_data in enumerate(training_data):
            if i % 50 == 0:
                print(f"Processing user {i+1}/{len(training_data)}")
            
            features = self.extract_ensemble_features(user_data)
            
            if feature_names is None:
                feature_names = sorted(features.keys())
            
            feature_vector = [features.get(name, 0.0) for name in feature_names]
            all_features.append(feature_vector)
        
        feature_matrix = np.array(all_features)
        print(f"Created feature matrix: {feature_matrix.shape}")
        
        return feature_matrix, feature_names
    
    def fit(self, training_data: List[Dict]) -> 'PRISMEnsembleModel':
        """Train the complete ensemble model - FIXED VERSION"""
        
        print("Training PRISM Ensemble Model...")
        print("=" * 50)
        
        # Step 1: Train component models COMPLETELY first
        print("Step 1: Training component models...")
        
        # Train Entropy Engine
        print("  Training Entropy Engine...")
        self.entropy_engine.fit(training_data)
        
        # Prepare data for Social Analyzer
        print("  Training Social Risk Analyzer...")
        users_data = []
        connections_data = []
        
        for user_data in training_data:
            user_profile = user_data.get('user_profile', {})
            user_profile['user_id'] = user_data.get('user_id')
            users_data.append(user_profile)
            
            for social_conn in user_data.get('social_data', []):
                connections_data.append({
                    'user_id': user_data.get('user_id'),
                    **social_conn
                })
        
        self.social_analyzer.build_social_graph(users_data, connections_data)
        self.social_analyzer.fit('default_probability')
        
        # Train Ghost Detector
        print("  Training Ghost Detector...")
        self.ghost_detector.fit(training_data)
        
        # Step 2: NOW extract features (after components are trained)
        print("Step 2: Extracting ensemble features...")
        X, feature_names = self.prepare_feature_matrix(training_data)
        
        # Scale features
        X_scaled = self.ensemble_scaler.fit_transform(X)
        
        # Step 3: Prepare targets
        print("Step 3: Preparing targets...")
        y_default = []
        y_risk_score = []
        
        for user_data in training_data:
            user_profile = user_data.get('user_profile', {})
            
            # Default classification target
            is_defaulted = 1 if user_profile.get('default_status') == 'defaulted' else 0
            y_default.append(is_defaulted)
            
            # Risk score regression target
            risk_score = user_profile.get('credit_risk_score', 600)
            y_risk_score.append(risk_score)
        
        y_default = np.array(y_default)
        y_risk_score = np.array(y_risk_score)
        
        print(f"Default rate in training data: {np.mean(y_default):.2%}")
        print(f"Average risk score: {np.mean(y_risk_score):.0f}")
        
        # Step 4: Train ensemble models
        print("Step 4: Training ensemble models...")
        
        # Split data for training/validation
        X_train, X_val, y_def_train, y_def_val, y_score_train, y_score_val = train_test_split(
            X_scaled, y_default, y_risk_score, test_size=0.2, random_state=42, stratify=y_default
        )
        
        # Train default classifier
        print("  Training default classifier...")
        self.default_classifier.fit(X_train, y_def_train)
        
        # Train risk score regressor
        print("  Training risk score regressor...")
        self.risk_score_regressor.fit(X_train, y_score_train)
        
        # Step 5: Evaluate performance
        print("Step 5: Evaluating performance...")
        
        # Default classification metrics
        y_def_pred = self.default_classifier.predict(X_val)
        y_def_proba = self.default_classifier.predict_proba(X_val)[:, 1]
        
        try:
            default_auc = roc_auc_score(y_def_val, y_def_proba)
        except:
            default_auc = 0.5
        
        # Risk score regression metrics
        y_score_pred = self.risk_score_regressor.predict(X_val)
        score_r2 = r2_score(y_score_val, y_score_pred)
        score_rmse = np.sqrt(mean_squared_error(y_score_val, y_score_pred))
        
        # Store performance metrics
        self.performance_metrics = {
            'default_classification': {
                'auc': round(default_auc, 4),
                'accuracy': round(np.mean(y_def_pred == y_def_val), 4)
            },
            'risk_score_regression': {
                'r2': round(score_r2, 4),
                'rmse': round(score_rmse, 2)
            }
        }
        
        # Feature importance
        self.feature_importance = {
            'default_features': dict(zip(feature_names, self.default_classifier.feature_importances_)),
            'risk_score_features': dict(zip(feature_names, self.risk_score_regressor.feature_importances_))
        }
        
        print("\nTraining Results:")
        print(f"  Default Classification AUC: {default_auc:.4f}")
        print(f"  Default Classification Accuracy: {np.mean(y_def_pred == y_def_val):.4f}")
        print(f"  Risk Score R²: {score_r2:.4f}")
        print(f"  Risk Score RMSE: {score_rmse:.2f}")
        
        # Top feature importance
        print("\nTop 5 Important Features for Default Prediction:")
        default_importance = sorted(self.feature_importance['default_features'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in default_importance:
            print(f"  {feature}: {importance:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, user_data: Dict) -> Dict:
        """Generate comprehensive risk prediction for a user"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble model must be fitted before prediction")
        
        start_time = datetime.now()
        
        # Extract features
        features = self.extract_ensemble_features(user_data)
        feature_names = sorted(features.keys())
        feature_vector = np.array([features.get(name, 0.0) for name in feature_names]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.ensemble_scaler.transform(feature_vector)
        
        # Individual model predictions
        default_probability = float(self.default_classifier.predict_proba(feature_vector_scaled)[0][1])
        risk_score_predicted = float(self.risk_score_regressor.predict(feature_vector_scaled)[0])
        
        # Ensemble prediction (weighted combination)
        final_default_probability = (
            0.6 * default_probability +
            0.2 * (1.0 - features['composite_lsi']) +  # Lower stability = higher risk
            0.1 * features['social_risk_score'] +
            0.1 * features['anomaly_score']
        )
        
        # Final risk score (ensemble of predictions and adjustments)
        base_risk_score = max(300, min(850, risk_score_predicted))
        
        # Adjustments based on component scores
        entropy_adjustment = (features['composite_lsi'] - 0.5) * 50  # -25 to +25 points
        social_adjustment = -(features['social_risk_score'] - 0.3) * 40  # Up to -28 to +12 points
        ghost_adjustment = -features['anomaly_score'] * 30  # Up to -30 points
        
        final_risk_score = int(max(300, min(850, 
            base_risk_score + entropy_adjustment + social_adjustment + ghost_adjustment
        )))
        
        # Risk category
        if final_risk_score >= 720:
            risk_category = "LOW"
        elif final_risk_score >= 600:
            risk_category = "MEDIUM"  
        elif final_risk_score >= 500:
            risk_category = "HIGH"
        else:
            risk_category = "VERY_HIGH"
        
        # Confidence score based on data completeness and model agreement
        confidence_score = features['data_completeness']
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return {
            'user_id': user_data.get('user_id'),
            'predictions': {
                'final_default_probability': round(final_default_probability, 4),
                'final_risk_score': final_risk_score,
                'risk_category': risk_category,
                'confidence_score': round(confidence_score, 4)
            },
            'component_predictions': {
                'entropy_lsi': round(features['composite_lsi'], 4),
                'social_risk_score': round(features['social_risk_score'], 4),
                'ghost_anomaly_score': round(features['anomaly_score'], 4),
                'default_classifier_proba': round(default_probability, 4),
                'risk_score_regressor': round(risk_score_predicted, 2)
            },
            'risk_factors': self._identify_risk_factors(features),
            'recommendations': self._generate_recommendations(features, final_default_probability, final_risk_score),
            'model_info': {
                'ensemble_weights': self.component_weights,
                'processing_time_ms': processing_time,
                'prediction_timestamp': datetime.now().isoformat(),
                'model_version': 'PRISM_v1.0'
            }
        }
    
    def _identify_risk_factors(self, features: Dict) -> List[Dict]:
        """Identify key risk factors from feature analysis"""
        
        risk_factors = []
        
        # High entropy (low stability)
        if features['composite_lsi'] < 0.3:
            risk_factors.append({
                'factor': 'LOW_LIFE_STABILITY',
                'severity': 'HIGH',
                'description': f"Life Stability Index ({features['composite_lsi']:.3f}) indicates unpredictable behavior patterns",
                'impact': 'Major negative factor'
            })
        
        # High social risk
        if features['social_risk_score'] > 0.6:
            risk_factors.append({
                'factor': 'HIGH_SOCIAL_RISK',
                'severity': 'MEDIUM',
                'description': f"Social network risk score ({features['social_risk_score']:.3f}) suggests risky connections",
                'impact': 'Moderate negative factor'
            })
        
        # Ghost/anomaly detection
        if features['anomaly_score'] > 0.7:
            risk_factors.append({
                'factor': 'SUSPICIOUS_BEHAVIOR_PATTERN',
                'severity': 'HIGH',
                'description': f"Behavioral anomaly detected (score: {features['anomaly_score']:.3f})",
                'impact': 'Requires investigation'
            })
        
        # Data quality issues
        if features['data_completeness'] < 0.3:
            risk_factors.append({
                'factor': 'INSUFFICIENT_DATA',
                'severity': 'MEDIUM',
                'description': f"Data completeness ({features['data_completeness']:.3f}) below recommended threshold",
                'impact': 'Reduces prediction confidence'
            })
        
        # Positive factors
        if features['composite_lsi'] > 0.7:
            risk_factors.append({
                'factor': 'HIGH_LIFE_STABILITY',
                'severity': 'POSITIVE',
                'description': f"High Life Stability Index ({features['composite_lsi']:.3f}) indicates reliable patterns",
                'impact': 'Major positive factor'
            })
        
        return risk_factors
    
    def _generate_recommendations(self, features: Dict, default_prob: float, risk_score: int) -> List[str]:
        """Generate actionable recommendations based on prediction"""
        
        recommendations = []
        
        # High risk recommendations
        if default_prob > 0.5 or risk_score < 500:
            recommendations.append("HIGH RISK: Enhanced due diligence required before credit approval")
            recommendations.append("Consider requiring additional collateral or guarantors")
            recommendations.append("Implement enhanced monitoring with monthly reviews")
        
        # Entropy-based recommendations
        if features['composite_lsi'] < 0.4:
            recommendations.append("High behavioral unpredictability detected - consider lifestyle stability incentives")
            recommendations.append("Recommend financial counseling or budgeting assistance")
        
        # Social risk recommendations
        if features['social_risk_score'] > 0.6:
            recommendations.append("High social network risk - monitor for cascade effects")
            recommendations.append("Consider diversifying borrower's social connections through community programs")
        
        # Data quality recommendations
        if features['data_completeness'] < 0.5:
            recommendations.append("Insufficient behavioral data - request additional transaction history")
            recommendations.append("Consider extending observation period before final decision")
        
        # Positive recommendations
        if default_prob < 0.15 and risk_score >= 700:
            recommendations.append("LOW RISK: Candidate for preferential rates and higher credit limits")
            recommendations.append("Consider for premium product offerings")
        
        return recommendations
    
    def get_model_performance(self) -> Dict:
        """Get comprehensive model performance metrics"""
        
        return {
            'ensemble_performance': self.performance_metrics,
            'feature_importance': {
                'top_default_features': sorted(
                    self.feature_importance['default_features'].items(),
                    key=lambda x: x[1], reverse=True
                )[:10],
                'top_risk_score_features': sorted(
                    self.feature_importance['risk_score_features'].items(),
                    key=lambda x: x[1], reverse=True
                )[:10]
            },
            'component_status': {
                'entropy_engine_fitted': self.entropy_engine.is_fitted,
                'social_analyzer_fitted': self.social_analyzer.is_fitted,
                'ghost_detector_fitted': self.ghost_detector.is_fitted,
                'ensemble_fitted': self.is_fitted
            }
        }
    
    def save_model(self, filepath: str):
        """Save the complete ensemble model"""
        
        model_data = {
            'entropy_engine': self.entropy_engine,
            'social_analyzer': self.social_analyzer,
            'ghost_detector': self.ghost_detector,
            'default_classifier': self.default_classifier,
            'risk_score_regressor': self.risk_score_regressor,
            'ensemble_scaler': self.ensemble_scaler,
            'component_weights': self.component_weights,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Complete PRISM Ensemble Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load complete ensemble model"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.entropy_engine = model_data['entropy_engine']
        self.social_analyzer = model_data['social_analyzer']  
        self.ghost_detector = model_data['ghost_detector']
        self.default_classifier = model_data['default_classifier']
        self.risk_score_regressor = model_data['risk_score_regressor']
        self.ensemble_scaler = model_data['ensemble_scaler']
        self.component_weights = model_data['component_weights']
        self.performance_metrics = model_data['performance_metrics']
        self.feature_importance = model_data['feature_importance']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Complete PRISM Ensemble Model loaded from {filepath}")
        return self