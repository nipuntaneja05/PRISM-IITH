"""
scripts/train_models.py
Complete PRISM Model Training Script
Trains all ML components and saves the ensemble model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from typing import List, Dict
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Import our ML components
from ml.ensemble_model import PRISMEnsembleModel

load_dotenv()

class PRISMTrainingPipeline:
    """Complete training pipeline for all PRISM components"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'prism_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
    
    def load_training_data(self) -> List[Dict]:
        """Load complete training data from database"""
        
        print("Loading training data from database...")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Load all users
            cursor.execute("SELECT user_id FROM user_profiles")
            user_ids = [row['user_id'] for row in cursor.fetchall()]
            
            print(f"Loading data for {len(user_ids)} users...")
            
            training_data = []
            
            for i, user_id in enumerate(user_ids):
                if i % 50 == 0:
                    print(f"  Loading user {i+1}/{len(user_ids)}")
                
                # Load user profile
                cursor.execute("""
                    SELECT user_id, age, gender, income_bracket, city, state,
                           credit_risk_score, default_probability, default_status,
                           composite_stability_score, created_at
                    FROM user_profiles WHERE user_id = %s
                """, (user_id,))
                user_profile = dict(cursor.fetchone())
                
                # Load movement data (last 90 days)
                cursor.execute("""
                    SELECT location_cluster_id, timestamp, stay_duration, 
                           location_lat, location_lng
                    FROM movement_patterns 
                    WHERE user_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 500
                """, (user_id,))
                movement_data = [dict(row) for row in cursor.fetchall()]
                
                # Load transaction data (last 90 days)
                cursor.execute("""
                    SELECT amount, category, payment_method, time_of_day,
                           timestamp, transaction_type, is_weekend
                    FROM transaction_streams 
                    WHERE user_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 200
                """, (user_id,))
                transaction_data = [dict(row) for row in cursor.fetchall()]
                
                # Load social connections
                cursor.execute("""
                    SELECT connected_user_id, connection_type, strength,
                           risk_correlation, frequency, total_transaction_amount
                    FROM social_connections 
                    WHERE user_id = %s
                """, (user_id,))
                social_data = [dict(row) for row in cursor.fetchall()]
                
                # Combine into user data structure
                user_data = {
                    'user_id': user_id,
                    'user_profile': user_profile,
                    'movement_data': movement_data,
                    'transaction_data': transaction_data,
                    'social_data': social_data
                }
                
                training_data.append(user_data)
            
            conn.close()
            
            print(f"Successfully loaded training data for {len(training_data)} users")
            print(f"Average records per user:")
            print(f"  Movement: {np.mean([len(u['movement_data']) for u in training_data]):.1f}")
            print(f"  Transactions: {np.mean([len(u['transaction_data']) for u in training_data]):.1f}")
            print(f"  Social: {np.mean([len(u['social_data']) for u in training_data]):.1f}")
            
            return training_data
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            return None
    
    def validate_data_quality(self, training_data: List[Dict]) -> bool:
        """Validate training data quality"""
        
        print("\nValidating training data quality...")
        
        if not training_data:
            print("❌ No training data loaded")
            return False
        
        # Check for defaulters in dataset
        defaulters = [u for u in training_data if u['user_profile'].get('default_status') == 'defaulted']
        default_rate = len(defaulters) / len(training_data)
        
        print(f"✓ Total users: {len(training_data)}")
        print(f"✓ Default rate: {default_rate:.2%} ({len(defaulters)} defaulters)")
        
        if default_rate < 0.05:
            print("⚠️  Warning: Low default rate may affect model performance")
        elif default_rate > 0.3:
            print("⚠️  Warning: Very high default rate - check data quality")
        
        # Check data completeness
        users_with_movement = len([u for u in training_data if u['movement_data']])
        users_with_transactions = len([u for u in training_data if u['transaction_data']])
        users_with_social = len([u for u in training_data if u['social_data']])
        
        print(f"✓ Users with movement data: {users_with_movement}/{len(training_data)} ({users_with_movement/len(training_data):.1%})")
        print(f"✓ Users with transaction data: {users_with_transactions}/{len(training_data)} ({users_with_transactions/len(training_data):.1%})")
        print(f"✓ Users with social data: {users_with_social}/{len(training_data)} ({users_with_social/len(training_data):.1%})")
        
        # Check for minimum data requirements
        if users_with_movement < len(training_data) * 0.8:
            print("⚠️  Warning: Many users lack sufficient movement data")
        
        if users_with_transactions < len(training_data) * 0.8:
            print("⚠️  Warning: Many users lack sufficient transaction data")
        
        print("✓ Data validation completed")
        return True
    
    def train_ensemble_model(self, training_data: List[Dict]) -> PRISMEnsembleModel:
        """Train the complete PRISM ensemble model"""
        
        print("\n" + "="*60)
        print("TRAINING PRISM ENSEMBLE MODEL")
        print("="*60)
        
        # Initialize ensemble model
        ensemble = PRISMEnsembleModel()
        
        # Train the complete ensemble
        try:
            ensemble.fit(training_data)
            print("\n✅ Ensemble model training completed successfully!")
            
            # Display performance metrics
            performance = ensemble.get_model_performance()
            print("\nModel Performance Summary:")
            print("-" * 40)
            
            ens_perf = performance['ensemble_performance']
            print(f"Default Classification AUC: {ens_perf['default_classification']['auc']:.4f}")
            print(f"Risk Score Regression R²: {ens_perf['risk_score_regression']['r2']:.4f}")
            print(f"Risk Score RMSE: {ens_perf['risk_score_regression']['rmse']:.2f}")
            print(f"Meta Classifier AUC: {ens_perf['meta_classifier']['auc']:.4f}")
            
            print("\nTop 5 Important Features:")
            for feature, importance in performance['feature_importance']['top_default_features'][:5]:
                print(f"  {feature}: {importance:.4f}")
            
            return ensemble
            
        except Exception as e:
            print(f"❌ Ensemble training failed: {e}")
            return None
    
    def save_models(self, ensemble: PRISMEnsembleModel):
        """Save all trained models"""
        
        print("\nSaving trained models...")
        
        try:
            # Save complete ensemble model
            ensemble.save_model('models/prism_ensemble_model.pkl')
            
            # Save individual components for API use
            ensemble.entropy_engine.save_model('models/entropy_engine.pkl')
            ensemble.social_analyzer.save_model('models/social_analyzer.pkl') 
            ensemble.ghost_detector.save_model('models/ghost_detector.pkl')
            
            # Save model metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'model_version': 'PRISM_v1.0',
                'performance_metrics': ensemble.get_model_performance(),
                'component_weights': ensemble.component_weights,
                'training_data_size': len(training_data) if 'training_data' in locals() else 0
            }
            
            import json
            with open('models/model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("✅ All models saved successfully!")
            print(f"📁 Models saved to: ./models/")
            print(f"🔧 Main ensemble model: prism_ensemble_model.pkl")
            print(f"📊 Metadata: model_metadata.json")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def test_model_predictions(self, ensemble: PRISMEnsembleModel, training_data: List[Dict]):
        """Test model predictions on sample data"""
        
        print("\nTesting model predictions...")
        
        # Test on first few users
        test_users = training_data[:5]
        
        for i, user_data in enumerate(test_users):
            try:
                prediction = ensemble.predict(user_data)
                
                actual_status = user_data['user_profile'].get('default_status', 'unknown')
                predicted_prob = prediction['predictions']['final_default_probability']
                predicted_score = prediction['predictions']['final_risk_score']
                
                print(f"\nUser {i+1} ({user_data['user_id']}):")
                print(f"  Actual Status: {actual_status}")
                print(f"  Predicted Default Prob: {predicted_prob:.4f}")
                print(f"  Predicted Risk Score: {predicted_score}")
                print(f"  Risk Category: {prediction['predictions']['risk_category']}")
                print(f"  Confidence: {prediction['predictions']['confidence_score']:.3f}")
                
            except Exception as e:
                print(f"Error testing user {i+1}: {e}")
        
        print("\nModel testing completed!")

def main():
    """Main training pipeline execution"""
    
    print("PRISM COMPLETE MODEL TRAINING PIPELINE")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    
    pipeline = PRISMTrainingPipeline()
    
    # Step 1: Load training data
    print("\nStep 1: Loading training data...")
    training_data = pipeline.load_training_data()
    
    if not training_data:
        print("Failed to load training data. Exiting.")
        return False
    
    # Step 2: Validate data quality
    print("\nStep 2: Validating data quality...")
    if not pipeline.validate_data_quality(training_data):
        print("Data validation failed. Exiting.")
        return False
    
    # Step 3: Train ensemble model
    print("\nStep 3: Training ensemble model...")
    ensemble = pipeline.train_ensemble_model(training_data)
    
    if not ensemble:
        print("Model training failed. Exiting.")
        return False
    
    # Step 4: Save models
    print("\nStep 4: Saving models...")
    pipeline.save_models(ensemble)
    
    # Step 5: Test predictions
    print("\nStep 5: Testing predictions...")
    pipeline.test_model_predictions(ensemble, training_data)
    
    # Final summary
    print("\n" + "=" * 50)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    performance = ensemble.get_model_performance()
    
    print("\nFINAL PERFORMANCE SUMMARY:")
    print(f"  Default Classification AUC: {performance['ensemble_performance']['default_classification']['auc']:.4f}")
    print(f"  Risk Score R-squared: {performance['ensemble_performance']['risk_score_regression']['r2']:.4f}")
    print(f"  Training Data Size: {len(training_data)} users")
    
    print(f"\nCOMPONENT STATUS:")
    status = performance['component_status']
    print(f"  Entropy Engine: {'✓ Trained' if status['entropy_engine_fitted'] else '✗ Failed'}")
    print(f"  Social Analyzer: {'✓ Trained' if status['social_analyzer_fitted'] else '✗ Failed'}")
    print(f"  Ghost Detector: {'✓ Trained' if status['ghost_detector_fitted'] else '✗ Failed'}")
    print(f"  Ensemble Model: {'✓ Trained' if status['ensemble_fitted'] else '✗ Failed'}")
    
    print(f"\nNEXT STEPS:")
    print("1. Update API to load trained models: Update src/api/main.py")
    print("2. Start API server: uvicorn src.api.main:app --reload --port 8000")
    print("3. Test endpoints: http://localhost:8000/docs")
    print("4. Frontend integration: Use API endpoints for real predictions")
    
    print(f"\nCompleted at: {datetime.now()}")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)