"""
Main Feature Extraction Script for Deepfake Audio Detection

Extract and save audio features for training or analysis.
Supports batch processing of audio datasets.

Usage:
    # Extract features from dataset
    python main_extract_features.py --input_dir ./data/raw --output_dir ./data/processed --feature_type cqcc
    
    # With labels CSV
    python main_extract_features.py --input_dir ./data/raw --output_dir ./data/processed --labels labels.csv
    
    # Multiple feature types
    python main_extract_features.py --input_dir ./data/raw --output_dir ./data/processed --feature_type cqcc mel mfcc
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.features.extract import extract_cqcc_features, extract_mel_spectrogram
import librosa
import numpy as np


def extract_features(
    audio_path: str,
    feature_type: str = 'mel',
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048
) -> np.ndarray:
    """
    Unified feature extraction function.
    
    Args:
        audio_path: Path to audio file
        feature_type: Type of features ('mel', 'cqcc', 'mfcc')
        sample_rate: Sample rate
        n_mels: Number of mel bands
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length
        n_fft: FFT window size
        
    Returns:
        Feature array
    """
    if feature_type == 'cqcc':
        return extract_cqcc_features(audio_path, sr=sample_rate)
    
    elif feature_type == 'mel':
        return extract_mel_spectrogram(
            audio_path, sr=sample_rate, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length
        )
    
    elif feature_type == 'mfcc':
        # Extract MFCCs using librosa
        y, sr = librosa.load(audio_path, sr=sample_rate)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length
        )
        return mfcc
    
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract Audio Features for Deepfake Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for extracted features')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels CSV file (optional)')
    
    # Feature extraction arguments
    parser.add_argument('--feature_type', type=str, nargs='+', default=['cqcc'],
                        choices=['mel', 'cqcc', 'mfcc'],
                        help='Feature type(s) to extract')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of mel bands')
    parser.add_argument('--n_mfcc', type=int, default=40,
                        help='Number of MFCC coefficients')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='Hop length for STFT')
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT window size')
    
    # Processing arguments
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
                        help='Audio file extensions to process')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search for audio files')
    parser.add_argument('--max_duration', type=float, default=None,
                        help='Maximum audio duration in seconds (clips longer audio)')
    
    # Output arguments
    parser.add_argument('--output_format', type=str, default='npy',
                        choices=['npy', 'npz', 'pt'],
                        help='Output file format for features')
    parser.add_argument('--save_metadata', action='store_true',
                        help='Save metadata JSON file')
    parser.add_argument('--create_manifest', action='store_true',
                        help='Create manifest CSV with file paths and labels')
    
    # Display arguments
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress bars')
    
    return parser.parse_args()


def find_audio_files(input_dir: str, extensions: List[str], recursive: bool = False) -> List[Path]:
    """
    Find all audio files in directory.
    
    Args:
        input_dir: Input directory path
        extensions: List of file extensions
        recursive: Search recursively
        
    Returns:
        List of audio file paths
    """
    input_path = Path(input_dir)
    audio_files = []
    
    # Create patterns for each extension
    patterns = [f"*.{ext}" for ext in extensions]
    
    if recursive:
        for pattern in patterns:
            audio_files.extend(input_path.rglob(pattern))
    else:
        for pattern in patterns:
            audio_files.extend(input_path.glob(pattern))
    
    return sorted(audio_files)


def load_labels(labels_path: str) -> dict:
    """
    Load labels from CSV file.
    
    Args:
        labels_path: Path to labels CSV
        
    Returns:
        Dictionary mapping filenames to labels
    """
    df = pd.read_csv(labels_path)
    
    # Assume CSV has 'filename' and 'label' columns
    if 'filename' not in df.columns or 'label' not in df.columns:
        raise ValueError("Labels CSV must have 'filename' and 'label' columns")
    
    return dict(zip(df['filename'], df['label']))


def extract_single_feature(
    audio_path: Path,
    feature_type: str,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    max_duration: Optional[float] = None
) -> np.ndarray:
    """
    Extract features from single audio file.
    
    Args:
        audio_path: Path to audio file
        feature_type: Type of features to extract
        sample_rate: Sample rate
        n_mels: Number of mel bands
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length for STFT
        n_fft: FFT window size
        max_duration: Maximum duration in seconds
        
    Returns:
        Feature array
    """
    try:
        features = extract_features(
            audio_path=str(audio_path),
            feature_type=feature_type,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            hop_length=hop_length,
            n_fft=n_fft
        )
        
        # Clip if max_duration specified
        if max_duration is not None:
            max_frames = int(max_duration * sample_rate / hop_length)
            if features.shape[-1] > max_frames:
                features = features[..., :max_frames]
        
        # Convert to numpy if needed
        if hasattr(features, 'cpu'):
            features = features.cpu().numpy()
        
        return features
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract features from {audio_path.name}: {e}")


def save_features(
    features: np.ndarray,
    output_path: Path,
    output_format: str = 'npy'
):
    """
    Save features to file.
    
    Args:
        features: Feature array
        output_path: Output file path
        output_format: Output format (npy, npz, pt)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'npy':
        np.save(output_path.with_suffix('.npy'), features)
    elif output_format == 'npz':
        np.savez_compressed(output_path.with_suffix('.npz'), features=features)
    elif output_format == 'pt':
        import torch
        torch.save(torch.from_numpy(features), output_path.with_suffix('.pt'))


def process_dataset(
    input_dir: str,
    output_dir: str,
    feature_types: List[str],
    labels_dict: Optional[dict] = None,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    max_duration: Optional[float] = None,
    extensions: List[str] = ['wav', 'mp3', 'flac'],
    recursive: bool = False,
    output_format: str = 'npy',
    save_metadata: bool = False,
    create_manifest: bool = False,
    quiet: bool = False
) -> dict:
    """
    Process entire dataset and extract features.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        feature_types: List of feature types to extract
        labels_dict: Optional dictionary of labels
        sample_rate: Sample rate
        n_mels: Number of mel bands
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length
        n_fft: FFT window size
        max_duration: Maximum duration
        extensions: File extensions to process
        recursive: Search recursively
        output_format: Output format
        save_metadata: Save metadata
        create_manifest: Create manifest CSV
        quiet: Suppress progress
        
    Returns:
        Dictionary with processing statistics
    """
    # Find audio files
    audio_files = find_audio_files(input_dir, extensions, recursive)
    
    if not audio_files:
        raise ValueError(f"No audio files found in {input_dir}")
    
    print(f"\nFound {len(audio_files)} audio files")
    print(f"Feature types: {', '.join(feature_types)}")
    print(f"Output directory: {output_dir}\n")
    
    # Statistics
    stats = {
        'total_files': len(audio_files),
        'processed': 0,
        'failed': 0,
        'feature_types': feature_types,
        'errors': []
    }
    
    # Manifest data
    manifest_data = []
    
    # Process each file
    iterator = tqdm(audio_files, desc="Extracting features", disable=quiet)
    
    for audio_path in iterator:
        try:
            # Get relative path to maintain directory structure
            rel_path = audio_path.relative_to(input_dir)
            
            # Extract features for each type
            for feature_type in feature_types:
                # Create output path
                output_subdir = Path(output_dir) / feature_type
                output_path = output_subdir / rel_path.with_suffix('')
                
                # Extract features
                features = extract_single_feature(
                    audio_path, feature_type, sample_rate,
                    n_mels, n_mfcc, hop_length, n_fft, max_duration
                )
                
                # Save features
                save_features(features, output_path, output_format)
                
                # Update progress
                if not quiet:
                    iterator.set_postfix({
                        'file': audio_path.name[:20],
                        'feature': feature_type,
                        'shape': str(features.shape)
                    })
            
            # Add to manifest
            if create_manifest:
                label = labels_dict.get(audio_path.name) if labels_dict else None
                manifest_data.append({
                    'filename': audio_path.name,
                    'relative_path': str(rel_path),
                    'label': label
                })
            
            stats['processed'] += 1
            
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append({
                'file': str(audio_path),
                'error': str(e)
            })
            print(f"\n⚠️  Error processing {audio_path.name}: {e}")
    
    # Save metadata
    if save_metadata:
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'input_dir': input_dir,
            'output_dir': output_dir,
            'feature_types': feature_types,
            'sample_rate': sample_rate,
            'n_mels': n_mels,
            'n_mfcc': n_mfcc,
            'hop_length': hop_length,
            'n_fft': n_fft,
            'max_duration': max_duration,
            'statistics': stats
        }
        
        metadata_path = Path(output_dir) / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Metadata saved: {metadata_path}")
    
    # Create manifest
    if create_manifest and manifest_data:
        manifest_df = pd.DataFrame(manifest_data)
        manifest_path = Path(output_dir) / 'manifest.csv'
        manifest_df.to_csv(manifest_path, index=False)
        print(f"✓ Manifest saved: {manifest_path}")
    
    return stats


def print_summary(stats: dict):
    """Print processing summary."""
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total files:      {stats['total_files']:>6}")
    print(f"Processed:        {stats['processed']:>6}")
    print(f"Failed:           {stats['failed']:>6}")
    print(f"Feature types:    {', '.join(stats['feature_types'])}")
    
    if stats['failed'] > 0:
        print(f"\n⚠️  {stats['failed']} files failed to process")
        if stats['errors']:
            print("\nFirst 5 errors:")
            for error in stats['errors'][:5]:
                print(f"  - {Path(error['file']).name}: {error['error']}")
    
    print("="*60)


def main():
    """Main feature extraction pipeline."""
    args = parse_args()
    
    # Print header
    if not args.quiet:
        print("\n" + "="*70)
        print(" " * 20 + "FEATURE EXTRACTION - DEEPFAKE AUDIO DETECTION")
        print("="*70)
        print(f"Input:  {args.input_dir}")
        print(f"Output: {args.output_dir}")
        print(f"Features: {', '.join(args.feature_type)}")
        print("="*70 + "\n")
    
    # Validate input directory
    if not Path(args.input_dir).exists():
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        return 1
    
    # Load labels if provided
    labels_dict = None
    if args.labels:
        try:
            labels_dict = load_labels(args.labels)
            print(f"✓ Loaded {len(labels_dict)} labels from {args.labels}\n")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load labels: {e}\n")
    
    # Process dataset
    try:
        stats = process_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            feature_types=args.feature_type,
            labels_dict=labels_dict,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_mfcc=args.n_mfcc,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            max_duration=args.max_duration,
            extensions=args.extensions,
            recursive=args.recursive,
            output_format=args.output_format,
            save_metadata=args.save_metadata,
            create_manifest=args.create_manifest,
            quiet=args.quiet
        )
        
        # Print summary
        if not args.quiet:
            print_summary(stats)
        
        if stats['failed'] > 0:
            print("\n⚠️  Some files failed to process. Check errors above.")
            return 1
        
        print("\n✅ Feature extraction completed successfully!\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
