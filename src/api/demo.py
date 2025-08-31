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
from typing import Optional, Dict, List

# --- Setup project root ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- TEMPORARY aliases so we can load the old pickle ---
import src.ml.entropy_engine as entropy_engine
import src.ml.social_risk_gnn as social_risk_gnn
import src.ml.ghost_detector as ghost_detector
import src.ml.ensemble_model as ensemble_model_mod

sys.modules['entropy_engine'] = entropy_engine
sys.modules['social_risk_gnn'] = social_risk_gnn
sys.modules['ghost_detector'] = ghost_detector
sys.modules['ensemble_model'] = ensemble_model_mod

# Load old pickle
with open("models/prism_ensemble_model.pkl", "rb") as f:
    old_model = pickle.load(f)

print("✓ Model loaded temporarily")

# --- Now save it again with the correct module paths ---
with open("models/prism_ensemble_model_repickled.pkl", "wb") as f:
    pickle.dump(old_model, f)

print("✓ Model re-pickled with updated imports")
