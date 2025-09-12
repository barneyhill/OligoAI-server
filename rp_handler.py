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

try:
    from rinalmo.data.alphabet import Alphabet
    from rinalmo.data.downstream.aso.dataset import ASODataset
    print("Successfully imported from RiNALMo")
except ImportError as e:
    print(f"Failed to import from RiNALMo: {e}")
    print(f"Python path: {sys.path}")
    print(f"Directory contents: {os.listdir('/workspace/RiNALMo') if os.path.exists('/workspace/RiNALMo') else 'RiNALMo not found'}")
    raise

# Import the model wrapper from the root of RiNALMo
try:
    from train_aso import ASOInhibitionPredictionWrapper
    print("Successfully imported ASOInhibitionPredictionWrapper")
except ImportError as e:
    print(f"Failed to import ASOInhibitionPredictionWrapper: {e}")
    print(f"Contents of /workspace/RiNALMo: {os.listdir('/workspace/RiNALMo') if os.path.exists('/workspace/RiNALMo') else 'Not found'}")
    # We'll need to handle this in the inference function
    ASOInhibitionPredictionWrapper = None

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
    
    # Add dummy columns that ASODataset expects but aren't needed for inference
    # These are only added internally - users don't need to provide them
    if 'split' not in df.columns:
        df['split'] = 'test'  # Dummy value required by ASODataset
    
    if 'inhibition_percent' not in df.columns:
        df['inhibition_percent'] = 0.0  # Dummy value required by ASODataset
    
    # Load the model
    print(f"Loading model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    
    if ASOInhibitionPredictionWrapper is None:
        # Fallback: try to load with pytorch lightning directly
        import lightning as pl
        try:
            model = pl.LightningModule.load_from_checkpoint(MODEL_PATH)
            print("Loaded model using Lightning fallback")
        except Exception as e:
            print(f"Failed to load model with Lightning: {e}")
            # Last resort: load state dict directly
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            print(f"Checkpoint keys: {checkpoint.keys()}")
            raise RuntimeError("Cannot load model without ASOInhibitionPredictionWrapper class")
    else:
        model = ASOInhibitionPredictionWrapper.load_from_checkpoint(MODEL_PATH)
        print(f"Model loaded successfully")
    
    model.eval()
    model.to(device)
    
    if hasattr(model, 'scaler'):
        print(f"Model has scaler: {type(model.scaler)}")
    
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
                # Unpack batch - ASODataset returns 8 items
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
                if hasattr(model, 'scaler') and model.scaler is not None:
                    unscaled_predictions = model.scaler.inverse_transform(scaled_predictions)
                else:
                    unscaled_predictions = scaled_predictions
                    
                all_predictions.extend(unscaled_predictions.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {(batch_idx + 1) * batch_size} samples...")
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
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
        "csv_data": "aso_sequence_5_to_3,rna_context,sugar_mods,...\\n...",  # Raw CSV string
        "batch_size": 32  # Optional, default 32
    }
    
    Required CSV columns:
    - aso_sequence_5_to_3
    - rna_context
    - sugar_mods (list format like "['MOE', 'MOE', 'DNA', ...]")
    - backbone_mods (list format like "['PS', 'PO', 'PS', ...]")
    - dosage
    - transfection_method
    - custom_id
    
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
            required_cols = ['aso_sequence_5_to_3', 'rna_context', 'sugar_mods', 'backbone_mods', 
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
    # RunPod serverless mode
    runpod.serverless.start({"handler": handler})
