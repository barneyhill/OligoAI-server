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

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.aso.dataset import ASODataset
from rinalmo.data.downstream.aso.train_aso import ASOInhibitionPredictionWrapper

MODEL_PATH = "/workspace/OligoAI_11_09_25.ckpt"


class ASODatasetWithLen(ASODataset):
    """Wrapper around ASODataset that adds __len__ method for DataLoader compatibility"""
    def __len__(self):
        return len(self.df)


def run_inference_on_csv_data(csv_data, batch_size=32, device="cuda"):
    """
    Run inference on CSV data and return predictions.
    
    Args:
        csv_data: Raw CSV string data
        batch_size: Batch size for inference
        device: Device to run on
        
    Returns:
        str: CSV string with idx and oligoai_score columns
    """
    
    # Parse the CSV data
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Store original indices
    df['original_idx'] = df.index
    
    # If 'split' column doesn't exist, add it with 'test' values
    if 'split' not in df.columns:
        df['split'] = 'test'
    
    # If 'inhibition_percent' doesn't exist, add dummy values
    if 'inhibition_percent' not in df.columns:
        df['inhibition_percent'] = 0.0  # Dummy values for inference
    
    # Load the model
    print(f"Loading model from: {MODEL_PATH}")
    model = ASOInhibitionPredictionWrapper.load_from_checkpoint(MODEL_PATH)
    model.eval()
    model.to(device)
    print(f"Model loaded successfully with scaler: {type(model.scaler)}")
    
    # Initialize alphabet
    alphabet = Alphabet()
    
    # Create a temporary file for the dataset (ASODataset needs a file path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        temp_path = tmp_file.name
        df.to_csv(tmp_file, index=False)
    
    try:
        # Create dataset
        dataset = ASODatasetWithLen(
            data_path=temp_path,
            alphabet=alphabet,
            pad_to_max_len=True,
        )
        
        print(f"Loaded dataset with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True if device == "cuda" else False,
            shuffle=False
        )
        
        # Run inference
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                aso_tokens, chem_tokens, backbone_tokens, context_tokens, _, dosage, transfection_method_tokens, _ = batch
                
                # Move tensors to device
                aso_tokens = aso_tokens.to(device)
                chem_tokens = chem_tokens.to(device)
                backbone_tokens = backbone_tokens.to(device)
                context_tokens = context_tokens.to(device)
                dosage = dosage.to(device)
                transfection_method_tokens = transfection_method_tokens.to(device)
                
                # Forward pass with mixed precision if on GPU
                if device == "cuda":
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        scaled_predictions = model(
                            aso_tokens, chem_tokens, backbone_tokens, 
                            context_tokens, dosage, transfection_method_tokens
                        )
                else:
                    scaled_predictions = model(
                        aso_tokens, chem_tokens, backbone_tokens, 
                        context_tokens, dosage, transfection_method_tokens
                    )
                
                # Inverse transform the predictions to get real values
                unscaled_predictions = model.scaler.inverse_transform(scaled_predictions)
                all_predictions.extend(unscaled_predictions.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {(batch_idx + 1) * batch_size} samples...")
        
    finally:
        # Clean up temp file
        os.unlink(temp_path)
    
    print(f"Inference completed. Generated {len(all_predictions)} predictions")
    
    # Create output dataframe with idx and oligoai_score
    output_df = pd.DataFrame({
        'idx': df['original_idx'].values,
        'oligoai_score': all_predictions
    })
    
    # Convert to CSV string
    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def handler(job):
    """
    RunPod handler that processes raw CSV data through the OligoAI model.
    
    Expected input:
    {
        "csv_data": "aso_sequence,chemistry,backbone,...\\n...",  # Raw CSV string
        "batch_size": 32  # Optional, default 32
    }
    
    Returns:
    {
        "csv_output": "idx,oligoai_score\\n0,85.3\\n1,72.1\\n..."
    }
    """
    
    try:
        job_input = job.get('input', {})
        
        # Get CSV data
        csv_data = job_input.get('csv_data')
        if not csv_data:
            return {"error": "No csv_data provided in input"}
        
        batch_size = job_input.get('batch_size', 32)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Validate CSV format
        try:
            # Quick validation that it's valid CSV
            test_df = pd.read_csv(io.StringIO(csv_data))
            num_rows = len(test_df)
            print(f"Received CSV with {num_rows} rows and columns: {list(test_df.columns)}")
            
            # Check for required columns
            required_cols = ['aso_sequence', 'chemistry', 'backbone', 'context_sequence', 
                           'dosage', 'transfection_method', 'custom_id']
            missing_cols = [col for col in required_cols if col not in test_df.columns]
            if missing_cols:
                return {"error": f"Missing required columns: {missing_cols}"}
                
        except Exception as e:
            return {"error": f"Invalid CSV format: {str(e)}"}
        
        # Run inference
        print("Starting inference...")
        csv_output = run_inference_on_csv_data(
            csv_data=csv_data, 
            batch_size=batch_size, 
            device=device
        )
        
        # Return the CSV output directly
        return {
            "csv_output": csv_output
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in handler: {error_trace}")
        return {
            "error": f"Failed to process CSV: {str(e)}",
            "traceback": error_trace
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
