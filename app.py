# FASTAPI application to serve the trained model
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib