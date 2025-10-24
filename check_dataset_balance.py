"""
Dataset Balance Checker
Analyzes the balance of REAL vs FAKE samples in the training dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def check_dataset_balance():
    """
    Check the balance of REAL vs FAKE samples in the dataset.
    
    Loads training labels, counts samples, calculates percentages,
    creates visualization, and warns about imbalance.
    """
    
    print("\n" + "="*70)
    print("  📊 DATASET BALANCE ANALYSIS")
    print("="*70)
    
    # Check if we have the raw audio files
    data_folder = Path('data')
    real_folder = data_folder / 'real'
    fake_folder = data_folder / 'fake'
    
    if not real_folder.exists() or not fake_folder.exists():
        print("\n❌ ERROR: Dataset folders not found!")
        print(f"   Expected: {real_folder} and {fake_folder}")
        return
    
    # Count audio files
    real_files = list(real_folder.glob('*.wav'))
    fake_files = list(fake_folder.glob('*.wav'))
    
    num_real = len(real_files)
    num_fake = len(fake_files)
    total = num_real + num_fake
    
    if total == 0:
        print("\n❌ ERROR: No audio files found in dataset!")
        return
    
    # Calculate percentages
    real_pct = (num_real / total) * 100
    fake_pct = (num_fake / total) * 100
    
    # Print results
    print(f"\n📁 Dataset Location: {data_folder}")
    print(f"\n📊 Sample Counts:")
    print(f"   Real: {num_real} samples ({real_pct:.1f}%)")
    print(f"   Fake: {num_fake} samples ({fake_pct:.1f}%)")
    print(f"   Total: {total} samples")
    
    # Check for imbalance
    print(f"\n⚖️  Balance Analysis:")
    imbalance_ratio = max(real_pct, fake_pct) / min(real_pct, fake_pct) if min(real_pct, fake_pct) > 0 else float('inf')
    
    if abs(real_pct - fake_pct) <= 10:
        print(f"   ✅ Well balanced! ({real_pct:.1f}% vs {fake_pct:.1f}%)")
        balance_status = "EXCELLENT"
    elif abs(real_pct - fake_pct) <= 20:
        print(f"   ✓ Acceptable balance ({real_pct:.1f}% vs {fake_pct:.1f}%)")
        balance_status = "GOOD"
    elif max(real_pct, fake_pct) > 70:
        print(f"   ⚠️  WARNING: Significant imbalance detected!")
        print(f"   Ratio: {real_pct:.1f}% vs {fake_pct:.1f}%")
        print(f"   This may cause bias towards the majority class!")
        balance_status = "IMBALANCED"
    else:
        print(f"   ⚠️  Moderate imbalance ({real_pct:.1f}% vs {fake_pct:.1f}%)")
        balance_status = "MODERATE"
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if max(real_pct, fake_pct) > 70:
        print(f"   • Add more {'FAKE' if real_pct > fake_pct else 'REAL'} samples")
        print(f"   • Target: ~{total//2} samples per class")
        print(f"   • Consider using class weights in training")
    elif total < 100:
        print(f"   • Dataset is small ({total} samples)")
        print(f"   • Recommended: At least 100-200 samples per class")
        print(f"   • Current performance may not generalize well")
    else:
        print(f"   • Dataset balance is good!")
        print(f"   • Continue with current training approach")
    
    # Create visualization
    print(f"\n📈 Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    categories = ['REAL\n(Human)', 'FAKE\n(AI)']
    counts = [num_real, num_fake]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Dataset Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax2.pie(
        counts, 
        labels=categories,
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax2.set_title('Percentage Distribution', fontsize=14, fontweight='bold')
    
    # Add balance status
    status_color = {
        'EXCELLENT': '#2ecc71',
        'GOOD': '#3498db',
        'MODERATE': '#f39c12',
        'IMBALANCED': '#e74c3c'
    }
    
    fig.suptitle(
        f'Dataset Balance: {balance_status}',
        fontsize=16,
        fontweight='bold',
        color=status_color.get(balance_status, 'black'),
        y=1.02
    )
    
    plt.tight_layout()
    
    # Save visualization
    output_file = 'dataset_balance.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: {output_file}")
    
    # Also show the plot
    plt.show()
    
    # Summary statistics
    print(f"\n" + "="*70)
    print(f"  📋 SUMMARY")
    print("="*70)
    print(f"  Total Samples: {total}")
    print(f"  Real Samples: {num_real} ({real_pct:.1f}%)")
    print(f"  Fake Samples: {num_fake} ({fake_pct:.1f}%)")
    print(f"  Balance Status: {balance_status}")
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if max(real_pct, fake_pct) > 70:
        print(f"\n  ⚠️  ACTION REQUIRED: Dataset is imbalanced!")
    else:
        print(f"\n  ✅ Dataset balance is acceptable")
    
    print("="*70)
    print()
    
    return {
        'total': total,
        'real': num_real,
        'fake': num_fake,
        'real_pct': real_pct,
        'fake_pct': fake_pct,
        'balance_status': balance_status,
        'imbalance_ratio': imbalance_ratio
    }


if __name__ == '__main__':
    result = check_dataset_balance()
    
    if result:
        print("\n✨ Analysis complete!")
        print(f"📊 View the visualization: dataset_balance.png")
