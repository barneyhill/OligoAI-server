# rp_handler.py
import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import runpod
import traceback
import io
import tempfile

# Add RiNALMo to path
sys.path.insert(0, '/workspace/RiNALMo')

# These imports are confirmed to work
from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.aso.dataset import ASODataset
from train_aso import ASOInhibitionPredictionWrapper

# --- ADD THIS ---
# Define the path to your model checkpoint
MODEL_PATH = "/workspace/OligoAI_11_09_25.ckpt"


def handler(job):
    """
    Tests if the model checkpoint can be loaded successfully.
    """
    
    # --- ADD THIS BLOCK ---
    try:
        if not os.path.exists(MODEL_PATH):
            return {"error": f"Model checkpoint not found at: {MODEL_PATH}"}
            
        print(f"Attempting to load model from: {MODEL_PATH}")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # This is the critical line to test
        model = ASOInhibitionPredictionWrapper.load_from_checkpoint(MODEL_PATH)
        model.to(device) # Also test moving the model to the device
        
        print("Model loaded successfully!")
        
        # Return a success message with some info
        return {
            "status": "SUCCESS",
            "message": "Model checkpoint loaded and moved to device successfully.",
            "model_class": str(type(model)),
            "device": str(model.device)
        }

    except Exception as e:
        # If it fails, return the full error and traceback
        error_trace = traceback.format_exc()
        print(f"Error loading model: {error_trace}")
        return {
            "status": "ERROR",
            "error": f"Failed to load model: {str(e)}",
            "traceback": error_trace
        }
# --- END OF ADDED BLOCK ---

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
