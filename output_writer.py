"""
Output writer module for gen-ai-docs CLI tool.
Handles writing generated markdown content to files.
"""

import os


def write_markdown_files(structured_data: dict, output_dir: str):
    """
    Write the generated markdown content to files in the specified directory.
    
    Args:
        structured_data (dict): Dictionary where keys are topic names and values are markdown content
        output_dir (str): Directory path where output files should be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write each topic to a separate markdown file
    for topic, content in structured_data.items():
        # Generate filename from topic name
        filename = topic.lower().replace(' ', '_').replace('&', 'and') + '.md'
        # Remove any other problematic characters for filenames
        filename = ''.join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
        
        # Construct full output path
        output_path = os.path.join(output_dir, filename)
        
        # Write content to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Saved: {output_path}")
        except Exception as e:
            print(f"  ✗ Failed to save {output_path}: {str(e)}") 