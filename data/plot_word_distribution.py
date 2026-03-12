import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

# Set style for a clean, modern look
sns.set_style("whitegrid", rc={'axes.edgecolor': '#E0E0E0', 'grid.color': '#F0F0F0'})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana']

# Get all .txt files in the current directory
txt_files = glob.glob("*.txt")

# Dictionary to store word counts per file
word_counts = {}

# Read each file and count words
for file_path in sorted(txt_files):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Split by whitespace and filter out empty strings
        words = [word for word in content.split() if word.strip()]
        word_count = len(words)
        
        # Remove .txt extension for legend
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        word_counts[file_name] = word_count

# Create DataFrame and sort by word count for better visualization
df = pd.DataFrame(list(word_counts.items()), columns=['File', 'Word_Count'])
df = df.sort_values('Word_Count', ascending=True)

# Create figure with better proportions
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#FAFAFA')

# Create a beautiful gradient color palette
# Using a sophisticated blue-to-purple-to-pink gradient
n = len(df)
colors = plt.cm.plasma(np.linspace(0.15, 0.85, n))

# Create horizontal bar plot with enhanced styling
bars = ax.barh(df['File'], df['Word_Count'], 
                color=colors, 
                edgecolor='white', 
                linewidth=2.5,
                height=0.75,
                alpha=0.9)

# Add subtle shadow effect by drawing slightly offset bars
for i, (bar, value) in enumerate(zip(bars, df['Word_Count'])):
    # Shadow effect
    shadow = Rectangle((0, bar.get_y() - 0.01), bar.get_width(), bar.get_height(),
                       facecolor='#CCCCCC', alpha=0.2, zorder=0, transform=ax.transData)
    ax.add_patch(shadow)

# Add value labels on bars with better styling
for i, (bar, value) in enumerate(zip(bars, df['Word_Count'])):
    width = bar.get_width()
    # Position label inside bar if there's enough space, otherwise outside
    if width > max(df['Word_Count']) * 0.15:
        ax.text(width * 0.98, bar.get_y() + bar.get_height()/2, 
                f'{value:,}', ha='right', va='center', 
                fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none'))
    else:
        ax.text(width + max(df['Word_Count']) * 0.015, bar.get_y() + bar.get_height()/2, 
                f'{value:,}', ha='left', va='center', 
                fontsize=11, fontweight='bold', color='#2C3E50')

# Enhanced styling
ax.set_xlabel('Number of Words', fontsize=14, fontweight='600', color='#2C3E50', labelpad=15)
ax.set_ylabel('', fontsize=14, fontweight='600', color='#2C3E50')
ax.set_title('Distribution of Number of Words Across All Files', 
             fontsize=18, fontweight='700', pad=30, color='#1A1A1A')

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#D0D0D0')
ax.spines['bottom'].set_linewidth(1.5)

# Enhanced grid styling
ax.grid(True, axis='x', alpha=0.4, linestyle='-', linewidth=0.8, color='#E8E8E8')
ax.set_axisbelow(True)

# Improve tick labels with better formatting
ax.tick_params(colors='#555555', labelsize=11, left=False)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['File'], fontsize=11, color='#2C3E50', fontweight='500')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Add subtle background color
ax.set_facecolor('#FFFFFF')

# Add padding for better spacing
ax.set_xlim(left=-max(df['Word_Count']) * 0.05, right=max(df['Word_Count']) * 1.15)

# Tight layout with extra padding
plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save the plot with high quality
plt.savefig('word_distribution.png', dpi=300, bbox_inches='tight', 
            facecolor='#FAFAFA', edgecolor='none', pad_inches=0.2)
print(f"Plot saved as 'word_distribution.png'")

# Also print summary statistics
print("\nSummary Statistics:")
print(df.describe())
print("\nWord counts per file:")
for file, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{file}: {count:,} words")

# Show the plot
plt.show()

