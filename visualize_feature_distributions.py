"""
Feature Quality Inspector
Analyzes CQCC feature separability between REAL and FAKE audio samples
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
from scipy import stats
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

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
        print(f"Error processing {audio_path}: {e}")
        return None

def visualize_feature_distributions():
    """
    Visualize and analyze feature separability between REAL and FAKE samples.
    
    Extracts CQCC features, computes statistics, performs statistical tests,
    and creates visualizations to assess feature quality.
    """
    
    print("\n" + "="*70)
    print("  🔬 FEATURE QUALITY ANALYSIS")
    print("="*70)
    
    # Setup paths
    data_folder = Path('data')
    real_folder = data_folder / 'real'
    fake_folder = data_folder / 'fake'
    
    # Get files
    real_files = sorted(list(real_folder.glob('*.wav')))[:50]
    fake_files = sorted(list(fake_folder.glob('*.wav')))[:50]
    
    print(f"\n📂 Loading samples:")
    print(f"   Real samples: {len(real_files)}")
    print(f"   Fake samples: {len(fake_files)}")
    
    # Extract features for REAL samples
    print("\n🎵 Extracting REAL audio features...")
    real_features = []
    for audio_file in tqdm(real_files, desc="Real files"):
        features = extract_cqcc_features(str(audio_file))
        if features is not None:
            # Flatten features for analysis
            real_features.append(features.flatten())
    
    # Extract features for FAKE samples
    print("\n🤖 Extracting FAKE audio features...")
    fake_features = []
    for audio_file in tqdm(fake_files, desc="Fake files"):
        features = extract_cqcc_features(str(audio_file))
        if features is not None:
            # Flatten features for analysis
            fake_features.append(features.flatten())
    
    real_features = np.array(real_features)
    fake_features = np.array(fake_features)
    
    print(f"\n📊 Feature shapes:")
    print(f"   Real: {real_features.shape}")
    print(f"   Fake: {fake_features.shape}")
    
    # Compute mean features
    real_mean = np.mean(real_features, axis=0)
    fake_mean = np.mean(fake_features, axis=0)
    real_std = np.std(real_features, axis=0)
    fake_std = np.std(fake_features, axis=0)
    
    print(f"\n📈 Statistical Summary:")
    print(f"   Real mean: {real_mean.mean():.4f} ± {real_std.mean():.4f}")
    print(f"   Fake mean: {fake_mean.mean():.4f} ± {fake_std.mean():.4f}")
    print(f"   Mean difference: {abs(real_mean.mean() - fake_mean.mean()):.4f}")
    
    # Perform t-tests on each feature dimension
    print(f"\n🧪 Statistical Testing (t-test)...")
    p_values = []
    separable_features = 0
    
    for i in range(min(real_features.shape[1], fake_features.shape[1])):
        t_stat, p_value = stats.ttest_ind(real_features[:, i], fake_features[:, i])
        p_values.append(p_value)
        if p_value < 0.05:
            separable_features += 1
    
    p_values = np.array(p_values)
    separability_pct = (separable_features / len(p_values)) * 100
    
    print(f"   Total feature dimensions: {len(p_values)}")
    print(f"   Separable features (p < 0.05): {separable_features} ({separability_pct:.1f}%)")
    print(f"   Mean p-value: {p_values.mean():.6f}")
    print(f"   Median p-value: {np.median(p_values):.6f}")
    
    # Calculate Jensen-Shannon divergence (distribution similarity)
    print(f"\n📏 Distribution Distance Analysis...")
    
    # Sample a subset of features for JS divergence calculation
    sample_indices = np.random.choice(real_features.shape[1], size=min(100, real_features.shape[1]), replace=False)
    js_divergences = []
    
    for idx in sample_indices:
        # Create histograms
        real_hist, _ = np.histogram(real_features[:, idx], bins=50, density=True)
        fake_hist, _ = np.histogram(fake_features[:, idx], bins=50, density=True)
        
        # Normalize to probability distributions
        real_hist = real_hist / (real_hist.sum() + 1e-10)
        fake_hist = fake_hist / (fake_hist.sum() + 1e-10)
        
        # Calculate JS divergence
        js_div = jensenshannon(real_hist, fake_hist)
        js_divergences.append(js_div)
    
    js_divergences = np.array(js_divergences)
    mean_js = js_divergences.mean()
    
    print(f"   Mean JS Divergence: {mean_js:.4f}")
    print(f"   JS Divergence range: [{js_divergences.min():.4f}, {js_divergences.max():.4f}]")
    
    # Interpretation
    print(f"\n💡 Feature Quality Assessment:")
    if mean_js < 0.1 and separability_pct < 20:
        print(f"   ❌ POOR: Features are too similar!")
        print(f"   → Real and fake samples are nearly indistinguishable")
        print(f"   → Feature extraction may need improvement")
        quality = "POOR"
        color = 'red'
    elif mean_js < 0.3 and separability_pct < 50:
        print(f"   ⚠️  MODERATE: Some separability but could be better")
        print(f"   → Features show some differences")
        print(f"   → Model may achieve moderate accuracy")
        quality = "MODERATE"
        color = 'orange'
    elif separability_pct >= 50:
        print(f"   ✅ GOOD: Features show clear separability!")
        print(f"   → {separability_pct:.1f}% of features are significantly different")
        print(f"   → Model should achieve good accuracy")
        quality = "GOOD"
        color = 'green'
    else:
        print(f"   ✓ ACCEPTABLE: Reasonable feature separability")
        quality = "ACCEPTABLE"
        color = 'blue'
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overlapping histograms for first 6 CQCC coefficients
    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        
        # Sample from features (take every Nth coefficient)
        coef_idx = i * (real_features.shape[1] // 6)
        
        ax.hist(real_features[:, coef_idx], bins=30, alpha=0.6, 
                color='green', label='REAL', density=True, edgecolor='black')
        ax.hist(fake_features[:, coef_idx], bins=30, alpha=0.6, 
                color='red', label='FAKE', density=True, edgecolor='black')
        
        # Calculate p-value for this coefficient
        t_stat, p_val = stats.ttest_ind(real_features[:, coef_idx], fake_features[:, coef_idx])
        
        # Highlight if separable
        border_color = 'green' if p_val < 0.05 else 'gray'
        ax.spines['top'].set_color(border_color)
        ax.spines['right'].set_color(border_color)
        ax.spines['bottom'].set_color(border_color)
        ax.spines['left'].set_color(border_color)
        ax.spines['top'].set_linewidth(3 if p_val < 0.05 else 1)
        ax.spines['right'].set_linewidth(3 if p_val < 0.05 else 1)
        ax.spines['bottom'].set_linewidth(3 if p_val < 0.05 else 1)
        ax.spines['left'].set_linewidth(3 if p_val < 0.05 else 1)
        
        ax.set_title(f'Coefficient {coef_idx}\n' + 
                     (f'p={p_val:.4f} ✓' if p_val < 0.05 else f'p={p_val:.4f}'),
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Feature Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # 2. P-value distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(p_values, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax7.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05 threshold')
    ax7.set_xlabel('P-value', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax7.set_title('P-value Distribution', fontsize=11, fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # 3. Mean feature comparison
    ax8 = fig.add_subplot(gs[2, 1])
    indices = np.arange(0, len(real_mean), len(real_mean) // 50)[:50]
    ax8.plot(indices, real_mean[indices], 'g-o', label='REAL', linewidth=2, markersize=4)
    ax8.plot(indices, fake_mean[indices], 'r-s', label='FAKE', linewidth=2, markersize=4)
    ax8.fill_between(indices, real_mean[indices] - real_std[indices], 
                      real_mean[indices] + real_std[indices], alpha=0.2, color='green')
    ax8.fill_between(indices, fake_mean[indices] - fake_std[indices], 
                      fake_mean[indices] + fake_std[indices], alpha=0.2, color='red')
    ax8.set_xlabel('Feature Index (sampled)', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Mean Feature Value', fontsize=10, fontweight='bold')
    ax8.set_title('Mean Feature Profiles', fontsize=11, fontweight='bold')
    ax8.legend()
    ax8.grid(alpha=0.3)
    
    # 4. Separability summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
    📊 SEPARABILITY SUMMARY
    
    Total Features: {len(p_values)}
    Separable (p<0.05): {separable_features}
    Separability: {separability_pct:.1f}%
    
    Mean JS Divergence: {mean_js:.4f}
    Mean P-value: {p_values.mean():.6f}
    
    Quality: {quality}
    
    {'✅ Features are well separated' if quality == 'GOOD' else ''}
    {'✓ Features show differences' if quality == 'ACCEPTABLE' else ''}
    {'⚠️ Limited separability' if quality == 'MODERATE' else ''}
    {'❌ Poor feature separation!' if quality == 'POOR' else ''}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle(f'Feature Separability Analysis - Quality: {quality}',
                 fontsize=16, fontweight='bold', color=color, y=0.98)
    
    # Save figure
    output_file = 'feature_separability.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: {output_file}")
    
    plt.show()
    
    # Final summary
    print(f"\n" + "="*70)
    print(f"  📋 ANALYSIS COMPLETE")
    print("="*70)
    print(f"  Feature Quality: {quality}")
    print(f"  Separable Features: {separable_features}/{len(p_values)} ({separability_pct:.1f}%)")
    print(f"  JS Divergence: {mean_js:.4f}")
    
    if quality == "POOR":
        print(f"\n  ⚠️  WARNING: Feature extraction may be problematic!")
        print(f"     → Real and fake features are too similar")
        print(f"     → Consider using different feature type")
        print(f"     → Check if synthetic data generation is too simplistic")
    
    print("="*70)
    print()
    
    return {
        'quality': quality,
        'separable_features': separable_features,
        'total_features': len(p_values),
        'separability_pct': separability_pct,
        'mean_js_divergence': mean_js,
        'mean_p_value': p_values.mean()
    }


if __name__ == '__main__':
    result = visualize_feature_distributions()
    
    if result:
        print("\n✨ Feature analysis complete!")
        print(f"📊 View the visualization: feature_separability.png")
