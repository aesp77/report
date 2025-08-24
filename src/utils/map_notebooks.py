# map_notebooks.py

import json
import os
import hashlib
from datetime import datetime

# Define the notebooks to process
notebooks = [
    "temporal_ae.ipynb",
    "temporal_gbo.ipynb", 
    "temporal.ipynb",
    "temporal_vae2.ipynb"
]

base_path = r"C:\Users\alessandro.esposito\source\repos\imperial\report\notebooks"

def create_cell_mapping(notebooks, base_path):
    """Create a map of identical cells across notebooks"""
    
    cell_hashes = {}
    
    for notebook_name in notebooks:
        notebook_path = os.path.join(base_path, notebook_name)
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                
                # Hash the cell content
                cell_hash = hashlib.md5(source.encode()).hexdigest()
                
                if cell_hash not in cell_hashes:
                    cell_hashes[cell_hash] = {
                        'content': source,  # Store full content
                        'preview': source[:200] + "..." if len(source) > 200 else source,
                        'notebooks': []
                    }
                
                cell_hashes[cell_hash]['notebooks'].append({
                    'name': notebook_name,
                    'cell_index': i
                })
    
    # Write results to file
    output_file = os.path.join(base_path, f"common_cells_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("COMMON CELLS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Notebooks analyzed: {', '.join(notebooks)}\n")
        f.write("="*80 + "\n\n")
        
        # Report cells that appear in multiple notebooks
        common_cells = []
        for hash_val, info in cell_hashes.items():
            if len(info['notebooks']) > 1:
                common_cells.append((hash_val, info))
        
        f.write(f"Found {len(common_cells)} cells appearing in multiple notebooks\n")
        f.write("-"*80 + "\n\n")
        
        for idx, (hash_val, info) in enumerate(common_cells, 1):
            f.write(f"COMMON CELL #{idx}\n")
            f.write(f"Hash: {hash_val}\n")
            f.write(f"Appears in {len(info['notebooks'])} notebooks:\n")
            
            for loc in info['notebooks']:
                f.write(f"  - {loc['name']}, Cell {loc['cell_index']}\n")
            
            f.write("\nContent Preview:\n")
            f.write("-"*40 + "\n")
            f.write(info['preview'])
            f.write("\n")
            
            f.write("\nFull Content:\n")
            f.write("-"*40 + "\n")
            f.write(info['content'])
            f.write("\n")
            f.write("="*80 + "\n\n")
        
        # Summary at the end
        f.write("\nSUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total unique cells: {len(cell_hashes)}\n")
        f.write(f"Cells appearing in multiple notebooks: {len(common_cells)}\n")
    
    print(f"Report saved to: {output_file}")
    
    # Also print summary to console
    print("\nSummary:")
    print(f"Total unique cells: {len(cell_hashes)}")
    print(f"Cells appearing in multiple notebooks: {len(common_cells)}")
    
    return cell_hashes, output_file

# Run the mapping
if __name__ == "__main__":
    mapping, report_file = create_cell_mapping(notebooks, base_path)
    print(f"\nCheck the detailed report at:\n{report_file}")