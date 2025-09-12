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
    raise

try:
    from train_aso import ASOInhibitionPredictionWrapper
    print("Successfully imported ASOInhibitionPredictionWrapper")
except ImportError as e:
    print(f"Failed to import ASOInhibitionPredictionWrapper: {e}")
    ASOInhibitionPredictionWrapper = None

MODEL_PATH = "/workspace/OligoAI_11_09_25.ckpt"


class ASODatasetWithLen(ASODataset):
    """Wrapper around ASODataset that adds __len__ method for DataLoader compatibility"""
    def __len__(self):
        return len(self.df)


def reverse_complement(seq):
    """Get reverse complement of RNA sequence (returns DNA complement)"""
    complement = {'A': 'T', 'U': 'A', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement.get(base, base) for base in seq[::-1])


def generate_aso_targets(target_rna, aso_length, sugar_mods, backbone_mods, dosage, transfection_method):
    """
    Generate all possible ASO sequences for a target RNA.
    
    Returns a DataFrame with all possible ASO positions and their sequences.
    """
    rows = []
    target_rna = target_rna.upper().replace('T', 'U')  # Ensure RNA format
    
    # Iterate through all possible positions
    for i in range(len(target_rna) - aso_length + 1):
        # Extract the target region and get reverse complement for ASO
        target_region = target_rna[i:i + aso_length]
        aso_seq = reverse_complement(target_region)
        
        # Extract RNA context (±50 nucleotides from ASO binding site)
        context_start = max(0, i - 50)
        context_end = min(len(target_rna), i + aso_length + 50)
        rna_context = target_rna[context_start:context_end]
        
        rows.append({
            'aso_sequence_5_to_3': aso_seq,
            'rna_context': rna_context,
            'sugar_mods': sugar_mods,
            'backbone_mods': backbone_mods,
            'dosage': dosage,
            'transfection_method': transfection_method,
            'custom_id': f'pos_{i}',  # Position in target RNA
            'position': i  # Store position for reference
        })
    
    return pd.DataFrame(rows)


def run_inference_on_generated_data(df, batch_size=32, device="cuda"):
    """Run inference on generated ASO DataFrame"""
    
    # Store original indices and positions
    df['original_idx'] = df.index
    original_positions = df['position'].values
    
    # Add dummy columns for ASODataset
    df['split'] = 'test'
    df['inhibition_percent'] = 0.0
    
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    
    if ASOInhibitionPredictionWrapper is None:
        import lightning as pl
        try:
            model = pl.LightningModule.load_from_checkpoint(MODEL_PATH)
        except Exception as e:
            raise RuntimeError("Cannot load model without ASOInhibitionPredictionWrapper class")
    else:
        model = ASOInhibitionPredictionWrapper.load_from_checkpoint(MODEL_PATH)
    
    model.eval()
    model.to(device)
    
    # Initialize alphabet
    alphabet = Alphabet()
    
    # Create temporary file for dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        temp_path = tmp_file.name
        df.to_csv(tmp_file, index=False)
    
    try:
        # Create dataset and dataloader
        dataset = ASODatasetWithLen(
            data_path=temp_path,
            alphabet=alphabet,
            pad_to_max_len=True,
        )
        
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
                
                # Move to device
                aso_tokens = aso_tokens.to(device)
                chem_tokens = chem_tokens.to(device)
                backbone_tokens = backbone_tokens.to(device)
                context_tokens = context_tokens.to(device)
                dosage = dosage.to(device)
                transfection_method_tokens = transfection_method_tokens.to(device)
                
                # Forward pass
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
                
                # Inverse transform predictions
                if hasattr(model, 'scaler') and model.scaler is not None:
                    unscaled_predictions = model.scaler.inverse_transform(scaled_predictions)
                else:
                    unscaled_predictions = scaled_predictions
                    
                all_predictions.extend(unscaled_predictions.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {(batch_idx + 1) * batch_size} samples...")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return all_predictions, original_positions


def handler(job):
    """
    RunPod handler that generates and scores all possible ASO sequences for a target RNA.
    
    Expected input:
    {
        "target_rna": "AUGCUGAUC...",  # Full RNA sequence
        "aso_length": 20,  # Length of ASO to generate
        "sugar_mods": "['MOE', 'MOE', 'DNA', ...]",  # Must match aso_length
        "backbone_mods": "['PS', 'PO', 'PS', ...]",  # Must match aso_length - 1
        "dosage": 1.0,
        "transfection_method": "gymnotic",
        "batch_size": 32  # Optional
    }
    
    Returns:
    {
        "csv_output": "position,aso_sequence,oligoai_score\\n0,ACGTACGT...,85.3\\n..."
    }
    """
    
    try:
        job_input = job.get('input', {})
        
        # Get inputs
        target_rna = job_input.get('target_rna')
        aso_length = job_input.get('aso_length')
        sugar_mods = job_input.get('sugar_mods')
        backbone_mods = job_input.get('backbone_mods')
        dosage = job_input.get('dosage')
        transfection_method = job_input.get('transfection_method')
        batch_size = job_input.get('batch_size', 32)
        
        # Validate inputs
        if not target_rna:
            return {"error": "No target_rna provided"}
        if not aso_length:
            return {"error": "No aso_length provided"}
        if len(target_rna) < aso_length + 100:
            return {"error": f"target_rna length ({len(target_rna)}) must be >= aso_length + 100 ({aso_length + 100}) to allow full context"}
        
        # Parse mods if they're strings
        if isinstance(sugar_mods, str):
            import ast
            sugar_mods = ast.literal_eval(sugar_mods)
        if isinstance(backbone_mods, str):
            import ast
            backbone_mods = ast.literal_eval(backbone_mods)
        
        # Validate mod lengths
        if len(sugar_mods) != aso_length:
            return {"error": f"sugar_mods length ({len(sugar_mods)}) must equal aso_length ({aso_length})"}
        if len(backbone_mods) != aso_length - 1:
            return {"error": f"backbone_mods length ({len(backbone_mods)}) must equal aso_length - 1 ({aso_length - 1})"}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Generate all ASO sequences
        print(f"Generating ASO sequences for RNA of length {len(target_rna)} with ASO length {aso_length}")
        df = generate_aso_targets(
            target_rna=target_rna,
            aso_length=aso_length,
            sugar_mods=str(sugar_mods),  # Convert back to string for DataFrame
            backbone_mods=str(backbone_mods),
            dosage=dosage,
            transfection_method=transfection_method
        )
        print(f"Generated {len(df)} ASO sequences")
        
        # Run inference
        print("Starting inference...")
        predictions, positions = run_inference_on_generated_data(
            df=df,
            batch_size=batch_size,
            device=device
        )
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'position': positions,
            'aso_sequence': df['aso_sequence_5_to_3'].values,
            'oligoai_score': predictions
        })
        
        # Sort by score (highest first)
        output_df = output_df.sort_values('oligoai_score', ascending=False)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)
        
        return {
            "csv_output": csv_buffer.getvalue(),
            "total_sequences": len(output_df),
            "best_position": int(output_df.iloc[0]['position']),
            "best_score": float(output_df.iloc[0]['oligoai_score'])
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in handler: {error_trace}")
        return {
            "error": f"Failed to process: {str(e)}",
            "traceback": error_trace
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
