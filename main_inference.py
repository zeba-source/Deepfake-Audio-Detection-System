"""Main inference script for deepfake audio detection."""
import argparse
import os
import sys
from datetime import datetime

from src.models.resnet_model import ResNetDeepfakeDetector
from src.models.multistream_model import MultiStreamDetector
from src.utils.inference import predict_audio, batch_predict


def log_message(message: str) -> None:
    """Print message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on audio file(s) for deepfake detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single audio file inference
  python main_inference.py --audio_path sample.wav --model_checkpoint models/checkpoints/best_model.pt

  # Batch inference on folder
  python main_inference.py --folder_path data/test_audio --model_checkpoint models/checkpoints/best_model.pt --output_csv results/predictions.csv

  # Use multi-stream model
  python main_inference.py --audio_path sample.wav --model_checkpoint models/checkpoints/best_model.pt --model_type multistream
        """
    )
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--audio_path', type=str,
                             help='Path to single audio file')
    input_group.add_argument('--folder_path', type=str,
                             help='Path to folder containing audio files')
    
    # Model arguments
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'multistream'],
                        help='Model architecture (default: resnet18)')
    
    # Feature arguments
    parser.add_argument('--feature_type', type=str, default='cqcc',
                        choices=['cqcc', 'mel'],
                        help='Feature type (default: cqcc)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sampling rate (default: 16000)')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV file for batch predictions (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run inference on (default: cpu)')
    
    args = parser.parse_args()
    
    log_message("=" * 60)
    log_message("DEEPFAKE AUDIO DETECTION - INFERENCE")
    log_message("=" * 60)
    
    # Validate checkpoint exists
    if not os.path.isfile(args.model_checkpoint):
        log_message(f"✗ Error: Checkpoint not found: {args.model_checkpoint}")
        sys.exit(1)
    
    # Initialize model
    log_message(f"Initializing {args.model_type} model...")
    try:
        if args.model_type == 'resnet18':
            model = ResNetDeepfakeDetector()
        else:
            model = MultiStreamDetector()
        log_message(f"✓ Model initialized")
    except Exception as e:
        log_message(f"✗ Model initialization failed: {e}")
        sys.exit(1)
    
    log_message(f"Device: {args.device}")
    log_message(f"Feature type: {args.feature_type}")
    log_message("")
    
    # Run inference
    try:
        if args.audio_path:
            # Single file inference
            log_message("Running single audio prediction...")
            log_message("-" * 60)
            
            if not os.path.isfile(args.audio_path):
                log_message(f"✗ Error: Audio file not found: {args.audio_path}")
                sys.exit(1)
            
            label, confidence = predict_audio(
                audio_path=args.audio_path,
                model=model,
                checkpoint_path=args.model_checkpoint,
                device=args.device,
                feature_type=args.feature_type,
                sample_rate=args.sample_rate
            )
            
            log_message("-" * 60)
            log_message("✓ Prediction completed successfully")
            
        else:
            # Batch inference
            log_message("Running batch predictions...")
            log_message("-" * 60)
            
            if not os.path.isdir(args.folder_path):
                log_message(f"✗ Error: Folder not found: {args.folder_path}")
                sys.exit(1)
            
            output_csv = args.output_csv if args.output_csv else 'results/batch_predictions.csv'
            
            results_df = batch_predict(
                folder_path=args.folder_path,
                model=model,
                checkpoint_path=args.model_checkpoint,
                output_csv=output_csv,
                device=args.device,
                feature_type=args.feature_type,
                sample_rate=args.sample_rate
            )
            
            log_message("-" * 60)
            log_message(f"✓ Batch predictions completed")
            log_message(f"✓ Results saved to {output_csv}")
            
            # Show preview of results
            if len(results_df) > 0:
                log_message("\nPreview of results:")
                print(results_df.head(10).to_string(index=False))
    
    except Exception as e:
        log_message(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    log_message("")
    log_message("=" * 60)
    log_message("INFERENCE COMPLETED SUCCESSFULLY!")
    log_message("=" * 60)


if __name__ == '__main__':
    main()
