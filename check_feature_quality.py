"""
Feature Quality Inspector
Analyzes CQCC feature distributions to verify they can separate REAL vs FAKE audio
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_cqcc_features(audio_path, sr=16000):
    """Extract CQCC features from audio file"""
    try:
        audio, _ = librosa.load(audio_path, sr=sr, duration=5.0)
        cqt = np.abs(librosa.cqt(audio, sr=sr, hop_length=512, n_bins=84))
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        target_frames = 256
        if cqt_db.shape[1] < target_frames:
            pad_width = target_frames - cqt_db.shape[1]
            cqt_db = np.pad(cqt_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            cqt_db = cqt_db[:, :target_frames]
        
        return cqt_db.T  # Shape: (time, freq)
    except Exception as e:
        return None

def calculate_kl_divergence(p, q, bins=50):
    """Calculate KL divergence between two distributions"""
    try:
        # Create histograms
        p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon
        
        # Normalize
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)
        
        # Calculate KL divergence
        kl_div = np.sum(p_hist * np.log(p_hist / q_hist))
        return kl_div
    except:
        return 0.0

def visualize_feature_distributions():
    """
    Analyze feature quality by comparing REAL vs FAKE distributions.
    """
    
    print("\n" + "="*70)
    print("  🔬 FEATURE QUALITY ANALYSIS")
    print("="*70)
    
    # Get file paths
    data_folder = Path('data')
    real_folder = data_folder / 'real'
    fake_folder = data_folder / 'fake'
    
    real_files = sorted(list(real_folder.glob('*.wav')))[:50]
    fake_files = sorted(list(fake_folder.glob('*.wav')))[:50]
    
    print(f"\n📂 Loading samples...")
    print(f"   Real samples: {len(real_files)}")
    print(f"   Fake samples: {len(fake_files)}")
    
    # Extract features
    print(f"\n🔧 Extracting CQCC features...")
    
    real_features_list = []
    print("   Processing REAL samples...")
    for audio_file in tqdm(real_files, desc="Real", leave=False):
        features = extract_cqcc_features(str(audio_file))
        if features is not None:
            real_features_list.append(features)
    
    fake_features_list = []
    print("   Processing FAKE samples...")
    for audio_file in tqdm(fake_files, desc="Fake", leave=False):
        features = extract_cqcc_features(str(audio_file))
        if features is not None:
            fake_features_list.append(features)
    
    print(f"   ✅ Extracted features from {len(real_features_list)} real + {len(fake_features_list)} fake samples")
    
    # Convert to arrays
    real_features = np.array(real_features_list)  # Shape: (n_samples, time, freq)
    fake_features = np.array(fake_features_list)
    
    print(f"\n📊 Feature dimensions:")
    print(f"   Real: {real_features.shape}")
    print(f"   Fake: {fake_features.shape}")
    
    # Flatten time dimension to get distribution across all frames
    real_flat = real_features.reshape(-1, real_features.shape[-1])  # (n_samples*time, freq)
    fake_flat = fake_features.reshape(-1, fake_features.shape[-1])
    
    # Compute mean features
    real_mean = np.mean(real_flat, axis=0)
    fake_mean = np.mean(fake_flat, axis=0)
    
    print(f"\n📈 Computing statistical separability...")
    
    # Statistical tests for each coefficient
    n_coeffs = real_flat.shape[1]
    p_values = []
    kl_divergences = []
    t_statistics = []
    
    for i in range(n_coeffs):
        # T-test
        t_stat, p_val = stats.ttest_ind(real_flat[:, i], fake_flat[:, i])
        p_values.append(p_val)
        t_statistics.append(abs(t_stat))
        
        # KL divergence
        kl_div = calculate_kl_divergence(real_flat[:, i], fake_flat[:, i])
        kl_divergences.append(kl_div)
    
    p_values = np.array(p_values)
    kl_divergences = np.array(kl_divergences)
    t_statistics = np.array(t_statistics)
    
    # Count separable coefficients
    separable = np.sum(p_values < 0.05)
    separable_pct = (separable / n_coeffs) * 100
    
    print(f"\n⚖️  Separability Analysis:")
    print(f"   Total coefficients: {n_coeffs}")
    print(f"   Significantly different (p < 0.05): {separable} ({separable_pct:.1f}%)")
    print(f"   Mean KL divergence: {np.mean(kl_divergences):.4f}")
    print(f"   Mean |t-statistic|: {np.mean(t_statistics):.2f}")
    
    # Quality assessment
    if separable_pct < 20:
        quality = "❌ POOR"
        warning = "Features are NOT separable! Feature extraction may be broken."
    elif separable_pct < 50:
        quality = "⚠️  MODERATE"
        warning = "Limited separability. Model may struggle to distinguish classes."
    elif separable_pct < 80:
        quality = "✓ GOOD"
        warning = "Good separability. Model should learn effectively."
    else:
        quality = "✅ EXCELLENT"
        warning = "Excellent separability! Features are highly discriminative."
    
    print(f"\n🎯 Feature Quality: {quality}")
    print(f"   {warning}")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Select top 12 most separable coefficients for visualization
    top_indices = np.argsort(kl_divergences)[-12:][::-1]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, coeff_idx in enumerate(top_indices):
        ax = axes[idx]
        
        # Plot histograms
        ax.hist(real_flat[:, coeff_idx], bins=50, alpha=0.6, color='green', 
                label='REAL', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(fake_flat[:, coeff_idx], bins=50, alpha=0.6, color='red', 
                label='FAKE', density=True, edgecolor='black', linewidth=0.5)
        
        # Highlight if significantly different
        if p_values[coeff_idx] < 0.05:
            ax.set_facecolor('#e8f5e9')  # Light green background
            significance = "✓"
        else:
            significance = "✗"
        
        # Title with statistics
        ax.set_title(f'Coeff {coeff_idx} {significance}\n'
                    f'KL={kl_divergences[coeff_idx]:.3f}, p={p_values[coeff_idx]:.4f}',
                    fontsize=9, fontweight='bold')
        ax.set_xlabel('Feature Value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Feature Separability Analysis - Top 12 Coefficients\n'
                 f'Quality: {quality} | {separable}/{n_coeffs} coefficients significant (p<0.05)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = 'feature_separability.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: {output_file}")
    plt.show()
    
    # Create summary plot
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Mean feature comparison
    ax1 = axes2[0]
    x = np.arange(n_coeffs)
    ax1.plot(x, real_mean, 'g-', label='REAL (mean)', linewidth=2)
    ax1.plot(x, fake_mean, 'r-', label='FAKE (mean)', linewidth=2)
    ax1.fill_between(x, real_mean, fake_mean, alpha=0.3, color='gray')
    ax1.set_xlabel('Coefficient Index', fontweight='bold')
    ax1.set_ylabel('Mean Feature Value', fontweight='bold')
    ax1.set_title('Mean CQCC Features Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: P-values
    ax2 = axes2[1]
    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    ax2.bar(x, -np.log10(p_values + 1e-300), color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=-np.log10(0.05), color='blue', linestyle='--', linewidth=2, label='p=0.05')
    ax2.set_xlabel('Coefficient Index', fontweight='bold')
    ax2.set_ylabel('-log10(p-value)', fontweight='bold')
    ax2.set_title(f'Statistical Significance\n({separable}/{n_coeffs} significant)', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: KL divergence
    ax3 = axes2[2]
    ax3.bar(x, kl_divergences, color='purple', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Coefficient Index', fontweight='bold')
    ax3.set_ylabel('KL Divergence', fontweight='bold')
    ax3.set_title(f'Feature Separability (KL Divergence)\nMean={np.mean(kl_divergences):.4f}', 
                  fontweight='bold')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    summary_file = 'feature_separability_summary.png'
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved summary to: {summary_file}")
    plt.show()
    
    # Final summary
    print(f"\n" + "="*70)
    print(f"  📋 SUMMARY")
    print("="*70)
    print(f"  Samples Analyzed: {len(real_features_list)} REAL + {len(fake_features_list)} FAKE")
    print(f"  Feature Dimensions: {real_features.shape[1]} × {real_features.shape[2]}")
    print(f"  Total Coefficients: {n_coeffs}")
    print(f"  Separable Coefficients: {separable} ({separable_pct:.1f}%)")
    print(f"  Mean KL Divergence: {np.mean(kl_divergences):.4f}")
    print(f"  Mean |t-statistic|: {np.mean(t_statistics):.2f}")
    print(f"\n  Quality Assessment: {quality}")
    print(f"  {warning}")
    
    if separable_pct < 20:
        print(f"\n  ⚠️  RECOMMENDATION:")
        print(f"     • Feature extraction may be broken!")
        print(f"     • Check if CQCC parameters are correct")
        print(f"     • Try different feature types (MFCC, Mel-spectrogram)")
    
    print("="*70)
    print()
    
    return {
        'n_coeffs': n_coeffs,
        'separable': separable,
        'separable_pct': separable_pct,
        'mean_kl': np.mean(kl_divergences),
        'mean_t_stat': np.mean(t_statistics),
        'quality': quality,
        'p_values': p_values,
        'kl_divergences': kl_divergences
    }


if __name__ == '__main__':
    result = visualize_feature_distributions()
    
    if result:
        print("\n✨ Feature quality analysis complete!")
        print(f"📊 View visualizations:")
        print(f"   • feature_separability.png (detailed)")
        print(f"   • feature_separability_summary.png (overview)")
