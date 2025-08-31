"""
PRISM Demo Runner - Complete setup and data generation
Run this after database setup to generate data and test the system
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_data import PRISMDataGenerator
import psycopg2
import json

load_dotenv()

def test_database_connection():
    """Test database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT', 5432)
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✓ Database connected: {version[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def generate_and_load_data():
    """Generate synthetic data and load into database"""
    print("Generating synthetic data for 500 users...")
    
    # Initialize generator
    generator = PRISMDataGenerator(num_users=500)
    
    # Generate all data
    stats = generator.generate_all_data()
    
    print(f"Generated data:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Save to CSV files for backup
    print("\nSaving data to CSV files...")
    generator.save_to_csv('data')
    
    # Insert into database
    print("\nInserting data into database...")
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 5432))
    }
    
    generator.insert_to_database(db_config)
    
    return True

def run_data_analysis():
    """Run basic data analysis to verify data quality"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT', 5432)
        )
        cursor = conn.cursor()
        
        print("\nData Analysis Summary:")
        print("=" * 50)
        
        # User distribution
        cursor.execute("""
            SELECT 
                risk_category, 
                COUNT(*) as count,
                ROUND(AVG(credit_risk_score), 0) as avg_score,
                ROUND(AVG(default_probability), 4) as avg_default_prob
            FROM user_profiles 
            GROUP BY risk_category 
            ORDER BY avg_default_prob DESC;
        """)
        
        print("Risk Category Distribution:")
        for row in cursor.fetchall():
            print(f"  {row[0]:12} | Count: {row[1]:3} | Avg Score: {row[2]:3.0f} | Default Prob: {row[3]:.4f}")
        
        # City distribution
        cursor.execute("""
            SELECT city, COUNT(*) as user_count 
            FROM user_profiles 
            GROUP BY city 
            ORDER BY user_count DESC 
            LIMIT 5;
        """)
        
        print("\nTop Cities by User Count:")
        for row in cursor.fetchall():
            print(f"  {row[0]:12} | Users: {row[1]}")
        
        # Transaction analysis
        cursor.execute("""
            SELECT 
                category,
                COUNT(*) as transaction_count,
                ROUND(AVG(amount), 2) as avg_amount
            FROM transaction_streams 
            GROUP BY category 
            ORDER BY transaction_count DESC 
            LIMIT 5;
        """)
        
        print("\nTop Transaction Categories:")
        for row in cursor.fetchall():
            print(f"  {row[0]:12} | Count: {row[1]:5} | Avg Amount: ₹{row[2]:8.2f}")
        
        # Movement patterns
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT user_id) as active_users,
                COUNT(*) as total_movements,
                ROUND(AVG(stay_duration), 1) as avg_stay_duration
            FROM movement_patterns;
        """)
        
        print("\nMovement Pattern Summary:")
        row = cursor.fetchone()
        print(f"  Active users with movement data: {row[0]}")
        print(f"  Total movement records: {row[1]:,}")
        print(f"  Average stay duration: {row[2]} minutes")
        
        # Social network analysis
        cursor.execute("""
            SELECT 
                COUNT(*) as total_connections,
                ROUND(AVG(strength), 3) as avg_strength,
                ROUND(AVG(risk_correlation), 3) as avg_risk_correlation
            FROM social_connections;
        """)
        
        print("\nSocial Network Summary:")
        row = cursor.fetchone()
        print(f"  Total connections: {row[0]:,}")
        print(f"  Average connection strength: {row[1]}")
        print(f"  Average risk correlation: {row[2]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error running data analysis: {e}")
        return False

def create_sample_api_responses():
    """Create sample data for API testing"""
    
    sample_responses = {
        "risk_score_response": {
            "user_id": "USR00001",
            "credit_risk_score": 680,
            "default_probability": 0.1250,
            "risk_category": "medium",
            "entropy_scores": {
                "movement_entropy": 0.45,
                "transaction_entropy": 0.52,
                "social_entropy": 0.38,
                "temporal_entropy": 0.41,
                "composite_score": 0.44
            },
            "factors": {
                "positive": [
                    "Stable movement patterns",
                    "Regular transaction frequency",
                    "Low-risk social connections"
                ],
                "negative": [
                    "High transaction entropy",
                    "Occasional unusual timing"
                ]
            },
            "last_updated": datetime.now().isoformat(),
            "processing_time_ms": 45
        },
        
        "social_risk_response": {
            "user_id": "USR00001",
            "direct_connections": 8,
            "cascade_risk_score": 0.23,
            "high_risk_connections": 1,
            "risk_propagation": {
                "1_hop": 0.15,
                "2_hop": 0.08,
                "3_hop": 0.03
            },
            "connection_analysis": [
                {
                    "connected_user": "USR00045",
                    "connection_strength": 0.85,
                    "risk_correlation": 0.67,
                    "impact": "medium"
                }
            ]
        },
        
        "entropy_calculation_response": {
            "user_id": "USR00001",
            "calculation_period": "90_days",
            "entropy_breakdown": {
                "movement": {
                    "value": 0.45,
                    "interpretation": "Moderately predictable movement",
                    "key_locations": 3,
                    "regularity_score": 0.78
                },
                "transaction": {
                    "value": 0.52,
                    "interpretation": "Variable spending patterns",
                    "categories_used": 8,
                    "temporal_consistency": 0.65
                },
                "social": {
                    "value": 0.38,
                    "interpretation": "Stable social network",
                    "active_connections": 6,
                    "interaction_regularity": 0.82
                },
                "temporal": {
                    "value": 0.41,
                    "interpretation": "Consistent daily routines",
                    "peak_hours": [9, 13, 19],
                    "weekend_variation": 0.15
                }
            },
            "composite_score": 0.44,
            "stability_rating": "MODERATE"
        }
    }
    
    # Save sample responses to file
    os.makedirs('sample_data', exist_ok=True)
    with open('sample_data/api_responses.json', 'w') as f:
        json.dump(sample_responses, f, indent=2)
    
    print("\n✓ Sample API responses saved to sample_data/api_responses.json")
    return True

def main():
    """Main demo runner function"""
    print("PRISM Credit Risk System - Demo Setup")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Check database connection
    print("Step 1: Testing database connection...")
    if not test_database_connection():
        print("✗ Please ensure database is running and credentials are correct")
        return False
    
    # Generate and load data
    print("\nStep 2: Generating and loading synthetic data...")
    if not generate_and_load_data():
        print("✗ Data generation failed")
        return False
    
    # Run data analysis
    print("\nStep 3: Running data quality analysis...")
    if not run_data_analysis():
        print("✗ Data analysis failed")
        return False
    
    # Create sample API responses
    print("\nStep 4: Creating sample API responses...")
    create_sample_api_responses()
    
    print("\n" + "=" * 60)
    print("✓ PRISM Demo Setup Complete!")
    print("\nSystem ready for:")
    print("  • ML model training")
    print("  • API development") 
    print("  • Frontend integration")
    print("  • Risk scoring demonstrations")
    
    print(f"\nGenerated data includes:")
    print(f"  • 500 diverse user profiles")
    print(f"  • 90 days of movement data")
    print(f"  • Transaction history")
    print(f"  • Social network connections")
    print(f"  • Behavioral fingerprints")
    
    print(f"\nNext steps:")
    print(f"  1. Train ML models: python scripts/train_models.py")
    print(f"  2. Start API server: uvicorn src.api.main:app --reload")
    print(f"  3. Test endpoints with sample_data/api_responses.json")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)