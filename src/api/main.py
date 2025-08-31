"""
src/api/main.py - Complete PRISM API with Statistics Endpoints
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import numpy as np

# Setup project root and imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "ml"))

# Import modules needed for pickle loading
import src.ml.entropy_engine
import src.ml.social_risk_gnn
import src.ml.ghost_detector
import src.ml.ensemble_model
# Add this at the top after imports
import logging

# Configure logging for Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Import with full paths for API use
from src.ml.ensemble_model import PRISMEnsembleModel

load_dotenv()

# Request Models
class UserRiskRequest(BaseModel):
    user_id: str

class BatchRiskRequest(BaseModel):
    user_ids: List[str]

# Global variables
trained_model: Optional[PRISMEnsembleModel] = None
db_params = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "prism_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

# def load_trained_model() -> bool:
#     """Load the trained ensemble model"""
#     global trained_model
    
#     model_path = project_root / "models" / "prism_ensemble_model.pkl"
    
#     if not model_path.exists():
#         print(f"Model file not found: {model_path}")
#         return False
    
#     try:
#         print(f"Loading model from {model_path}...")
#         with open(model_path, "rb") as f:
#             data = pickle.load(f)
        
#         # Handle both dict and object loading
#         if isinstance(data, dict):
#             print("Reconstructing model from dict...")
#             trained_model = PRISMEnsembleModel()
#             for key, value in data.items():
#                 setattr(trained_model, key, value)
#         else:
#             trained_model = data
        
#         fitted = getattr(trained_model, "is_fitted", False)
#         print(f"Model loaded successfully! Fitted: {fitted}")
#         return True
        
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# Update the model loading function
def load_trained_model() -> bool:
    """Load the trained ensemble model"""
    global trained_model
    
    model_path = project_root / "models" / "prism_ensemble_model.pkl"
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        
        # Handle both dict and object loading
        if isinstance(data, dict):
            logger.info("Reconstructing model from dict...")
            trained_model = PRISMEnsembleModel()
            for key, value in data.items():
                setattr(trained_model, key, value)
        else:
            trained_model = data
        
        fitted = getattr(trained_model, "is_fitted", False)
        logger.info(f"Model loaded successfully! Fitted: {fitted}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_db_connection():
    """Get database connection"""
    try:
        return psycopg2.connect(**db_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def get_user_data(user_id: str) -> Dict:
    """Get comprehensive user data from database"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # User profile
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
        user_profile = cursor.fetchone()
        
        if not user_profile:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        # Movement data (last 90 days)
        cursor.execute("""
            SELECT * FROM movement_patterns 
            WHERE user_id = %s AND timestamp >= %s 
            ORDER BY timestamp DESC LIMIT 500
        """, (user_id, datetime.now() - timedelta(days=90)))
        movement_data = cursor.fetchall()
        
        # Transaction data
        cursor.execute("""
            SELECT * FROM transaction_streams 
            WHERE user_id = %s AND timestamp >= %s 
            ORDER BY timestamp DESC LIMIT 300
        """, (user_id, datetime.now() - timedelta(days=90)))
        transaction_data = cursor.fetchall()
        
        # Social data
        cursor.execute("SELECT * FROM social_connections WHERE user_id = %s", (user_id,))
        social_data = cursor.fetchall()
        
        return {
            'user_id': user_id,
            'user_profile': dict(user_profile),
            'movement_data': [dict(row) for row in movement_data],
            'transaction_data': [dict(row) for row in transaction_data],
            'social_data': [dict(row) for row in social_data]
        }
        
    finally:
        conn.close()

# Load model on startup
print("=== PRISM API STARTUP ===")
model_loaded = load_trained_model()

# FastAPI app
app = FastAPI(
    title="PRISM Credit Risk API",
    description="Real-time credit risk assessment with trained ML models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add health check for database
@app.get("/api/v1/db-health")
async def database_health_check():
    """Check database connectivity"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {
            "database_status": "healthy",
            "connection_test": "passed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "database_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Basic endpoints
@app.get("/")
async def root():
    return {
        "message": "PRISM Credit Risk API",
        "status": "operational",
        "model_loaded": trained_model is not None,
        "model_fitted": trained_model.is_fitted if trained_model else False,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": trained_model is not None,
        "model_fitted": trained_model.is_fitted if trained_model else False,
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0"
    }

# Core prediction endpoints
@app.post("/api/v1/risk-score")
async def calculate_risk_score(request: UserRiskRequest):
    """Main risk scoring endpoint"""
    
    if not trained_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = datetime.now()
    user_data = get_user_data(request.user_id)
    
    try:
        prediction = trained_model.predict(user_data)
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        prediction['model_info']['processing_time_ms'] = processing_time
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/v1/user/{user_id}/dashboard")
async def get_user_dashboard(user_id: str):
    """User dashboard endpoint"""
    
    if not trained_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    user_data = get_user_data(user_id)
    prediction = trained_model.predict(user_data)
    
    return {
        "user_id": user_id,
        "risk_assessment": {
            "risk_score": prediction['predictions']['final_risk_score'],
            "risk_category": prediction['predictions']['risk_category'],
            "default_probability": prediction['predictions']['final_default_probability'],
            "confidence": prediction['predictions']['confidence_score']
        },
        "component_analysis": prediction['component_predictions'],
        "risk_factors": prediction['risk_factors'][:5],
        "recommendations": prediction['recommendations'][:3],
        "data_summary": {
            "movement_records": len(user_data['movement_data']),
            "transaction_records": len(user_data['transaction_data']),
            "social_connections": len(user_data['social_data'])
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/batch/risk-scores")
async def batch_risk_scores(request: BatchRiskRequest):
    """Batch risk scoring for multiple users"""
    
    if not trained_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    for user_id in request.user_ids[:20]:  # Limit to 20 users
        try:
            user_data = get_user_data(user_id)
            prediction = trained_model.predict(user_data)
            results.append({
                "user_id": user_id,
                "risk_score": prediction['predictions']['final_risk_score'],
                "default_probability": prediction['predictions']['final_default_probability'],
                "risk_category": prediction['predictions']['risk_category'],
                "confidence": prediction['predictions']['confidence_score']
            })
        except Exception as e:
            results.append({"user_id": user_id, "error": str(e)})
    
    return {
        "batch_results": results,
        "processed_count": len(results),
        "timestamp": datetime.now().isoformat()
    }

# Statistics endpoints for webpage
@app.get("/api/v1/statistics/overview")
async def get_system_overview():
    """System overview statistics"""
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # User statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_users,
                COUNT(CASE WHEN default_status = 'defaulted' THEN 1 END) as defaulted_users,
                AVG(credit_risk_score) as avg_risk_score,
                AVG(default_probability) as avg_default_probability,
                COUNT(CASE WHEN risk_category = 'HIGH' THEN 1 END) as high_risk_users,
                COUNT(CASE WHEN risk_category = 'VERY_HIGH' THEN 1 END) as very_high_risk_users
            FROM user_profiles
        """)
        user_stats = cursor.fetchone()
        
        # Activity statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_transactions,
                AVG(amount) as avg_transaction_amount,
                COUNT(DISTINCT user_id) as active_users_90d
            FROM transaction_streams 
            WHERE timestamp >= NOW() - INTERVAL '90 days'
        """)
        activity_stats = cursor.fetchone()
        
        # Social network statistics
        cursor.execute("SELECT COUNT(*) as total_connections FROM social_connections")
        social_stats = cursor.fetchone()
        
        # Movement statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_movement_records,
                COUNT(DISTINCT user_id) as users_with_movement
            FROM movement_patterns
        """)
        movement_stats = cursor.fetchone()
        
        return {
            "system_overview": {
                "total_users": user_stats['total_users'],
                "defaulted_users": user_stats['defaulted_users'],
                "default_rate": round(user_stats['defaulted_users'] / user_stats['total_users'] * 100, 2),
                "avg_risk_score": round(user_stats['avg_risk_score'], 0),
                "high_risk_users": user_stats['high_risk_users'],
                "very_high_risk_users": user_stats['very_high_risk_users']
            },
            "activity_stats": {
                "total_transactions": activity_stats['total_transactions'],
                "avg_transaction_amount": round(activity_stats['avg_transaction_amount'], 2),
                "active_users_90d": activity_stats['active_users_90d']
            },
            "data_coverage": {
                "total_social_connections": social_stats['total_connections'],
                "total_movement_records": movement_stats['total_movement_records'],
                "users_with_movement": movement_stats['users_with_movement']
            },
            "model_info": {
                "model_loaded": trained_model is not None,
                "model_fitted": trained_model.is_fitted if trained_model else False
            },
            "timestamp": datetime.now().isoformat()
        }
        
    finally:
        conn.close()

@app.get("/api/v1/statistics/risk-distribution")
async def get_risk_distribution():
    """Risk score distribution statistics"""
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Risk category distribution
        cursor.execute("""
            SELECT 
                risk_category,
                COUNT(*) as count,
                ROUND(AVG(credit_risk_score), 0) as avg_score,
                ROUND(AVG(default_probability), 4) as avg_default_prob
            FROM user_profiles 
            GROUP BY risk_category 
            ORDER BY avg_default_prob DESC
        """)
        risk_categories = cursor.fetchall()
        
        # Risk score ranges
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN credit_risk_score >= 750 THEN 'Excellent (750+)'
                    WHEN credit_risk_score >= 700 THEN 'Good (700-749)'
                    WHEN credit_risk_score >= 650 THEN 'Fair (650-699)'
                    WHEN credit_risk_score >= 600 THEN 'Poor (600-649)'
                    ELSE 'Very Poor (<600)'
                END as score_range,
                COUNT(*) as count,
                MIN(credit_risk_score) as min_score,
                MAX(credit_risk_score) as max_score
            FROM user_profiles 
            GROUP BY score_range
            ORDER BY min_score DESC
        """)
        score_ranges = cursor.fetchall()
        
        # City-wise risk distribution
        cursor.execute("""
            SELECT 
                city,
                COUNT(*) as users,
                ROUND(AVG(credit_risk_score), 0) as avg_risk_score,
                COUNT(CASE WHEN default_status = 'defaulted' THEN 1 END) as defaulters
            FROM user_profiles 
            GROUP BY city 
            ORDER BY defaulters DESC, users DESC
            LIMIT 10
        """)
        city_distribution = cursor.fetchall()
        
        return {
            "risk_categories": [dict(row) for row in risk_categories],
            "score_ranges": [dict(row) for row in score_ranges],
            "city_distribution": [dict(row) for row in city_distribution],
            "timestamp": datetime.now().isoformat()
        }
        
    finally:
        conn.close()

@app.get("/api/v1/statistics/model-performance")
async def get_model_performance_stats():
    """Model performance statistics"""
    
    if not trained_model:
        return {"error": "Model not loaded"}
    
    try:
        performance = trained_model.get_model_performance()
        
        # Add some computed metrics
        ensemble_perf = performance['ensemble_performance']
        
        return {
            "model_performance": {
                "default_classification_auc": ensemble_perf['default_classification']['auc'],
                "default_classification_accuracy": ensemble_perf['default_classification']['accuracy'],
                "risk_score_r2": ensemble_perf['risk_score_regression']['r2'],
                "risk_score_rmse": ensemble_perf['risk_score_regression']['rmse'],
                "performance_grade": "Excellent" if ensemble_perf['default_classification']['auc'] > 0.8 else 
                                  "Good" if ensemble_perf['default_classification']['auc'] > 0.7 else "Fair"
            },
            "component_status": performance['component_status'],
            "top_features": performance['feature_importance']['top_default_features'][:10],
            "model_info": {
                "components_trained": sum(performance['component_status'].values()),
                "total_components": len(performance['component_status'])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Failed to get performance stats: {str(e)}"}

@app.get("/api/v1/statistics/activity-trends")
async def get_activity_trends():
    """Activity trends over time"""
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Daily transaction trends (last 30 days)
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as transaction_count,
                AVG(amount) as avg_amount,
                COUNT(DISTINCT user_id) as active_users
            FROM transaction_streams 
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        """)
        daily_trends = cursor.fetchall()
        
        # Transaction category breakdown
        cursor.execute("""
            SELECT 
                category,
                COUNT(*) as count,
                AVG(amount) as avg_amount,
                SUM(amount) as total_amount
            FROM transaction_streams 
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY category 
            ORDER BY count DESC
            LIMIT 10
        """)
        category_breakdown = cursor.fetchall()
        
        # Payment method distribution
        cursor.execute("""
            SELECT 
                payment_method,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM transaction_streams 
            WHERE timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY payment_method 
            ORDER BY count DESC
        """)
        payment_methods = cursor.fetchall()
        
        return {
            "daily_trends": [dict(row) for row in daily_trends],
            "category_breakdown": [dict(row) for row in category_breakdown],
            "payment_methods": [dict(row) for row in payment_methods],
            "analysis_period": "30 days",
            "timestamp": datetime.now().isoformat()
        }
        
    finally:
        conn.close()

@app.get("/api/v1/statistics/social-network")
async def get_social_network_stats():
    """Social network analysis statistics"""
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Network overview
        cursor.execute("""
            SELECT 
                COUNT(*) as total_connections,
                AVG(strength) as avg_connection_strength,
                AVG(risk_correlation) as avg_risk_correlation,
                COUNT(CASE WHEN strength > 0.7 THEN 1 END) as strong_connections,
                COUNT(DISTINCT user_id) as users_with_connections
            FROM social_connections
        """)
        network_overview = cursor.fetchone()
        
        # Connection type distribution
        cursor.execute("""
            SELECT 
                connection_type,
                COUNT(*) as count,
                AVG(strength) as avg_strength,
                AVG(risk_correlation) as avg_risk_correlation
            FROM social_connections 
            GROUP BY connection_type 
            ORDER BY count DESC
        """)
        connection_types = cursor.fetchall()
        
        # High-risk network analysis
        cursor.execute("""
            SELECT 
                sc.user_id,
                up.risk_category,
                COUNT(*) as connections,
                AVG(sc.risk_correlation) as avg_risk_correlation,
                COUNT(CASE WHEN up2.default_status = 'defaulted' THEN 1 END) as defaulted_connections
            FROM social_connections sc
            JOIN user_profiles up ON sc.user_id = up.user_id
            JOIN user_profiles up2 ON sc.connected_user_id = up2.user_id
            GROUP BY sc.user_id, up.risk_category
            HAVING COUNT(CASE WHEN up2.default_status = 'defaulted' THEN 1 END) > 0
            ORDER BY defaulted_connections DESC, avg_risk_correlation DESC
            LIMIT 10
        """)
        high_risk_networks = cursor.fetchall()
        
        return {
            "network_overview": dict(network_overview),
            "connection_types": [dict(row) for row in connection_types],
            "high_risk_networks": [dict(row) for row in high_risk_networks],
            "network_health": {
                "density": round(network_overview['total_connections'] / network_overview['users_with_connections'], 2) if network_overview['users_with_connections'] > 0 else 0,
                "avg_strength": round(network_overview['avg_connection_strength'], 3),
                "risk_correlation": round(network_overview['avg_risk_correlation'], 3)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    finally:
        conn.close()

# Test and utility endpoints
@app.get("/api/v1/test/{user_id}")
async def test_prediction(user_id: str):
    """Quick test endpoint"""
    
    if not trained_model:
        return {"error": "Model not loaded"}
    
    try:
        user_data = get_user_data(user_id)
        prediction = trained_model.predict(user_data)
        
        return {
            "user_id": user_id,
            "risk_score": prediction['predictions']['final_risk_score'],
            "risk_category": prediction['predictions']['risk_category'],
            "default_probability": prediction['predictions']['final_default_probability'],
            "confidence": prediction['predictions']['confidence_score'],
            "data_records": {
                "movement": len(user_data['movement_data']),
                "transactions": len(user_data['transaction_data']),
                "social": len(user_data['social_data'])
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/users/sample")
async def get_sample_users():
    """Get sample user IDs for testing"""
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, risk_category, credit_risk_score 
            FROM user_profiles 
            ORDER BY RANDOM() 
            LIMIT 20
        """)
        sample_users = cursor.fetchall()
        
        return {
            "sample_users": [
                {
                    "user_id": row[0], 
                    "risk_category": row[1], 
                    "risk_score": row[2]
                } for row in sample_users
            ],
            "count": len(sample_users)
        }
        
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)