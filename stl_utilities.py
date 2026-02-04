"""
STL Headform Integration Utility

This script processes NIOSH STL headform files and extracts reference measurements
for integration with the mask fitting application.
"""

import struct
import numpy as np
import json
from pathlib import Path

def read_stl_ascii(filename):
    """Read ASCII STL file and extract vertices"""
    vertices = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('vertex'):
                # Parse vertex line: "vertex x y z"
                parts = line.split()
                if len(parts) == 4:
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    vertices.append(vertex)
    
    return np.array(vertices)

def calculate_headform_measurements(vertices):
    """Calculate key facial measurements from headform vertices"""
    
    # Get bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    # Calculate key measurements
    measurements = {
        'face_width': max_coords[0] - min_coords[0],  # X axis (left-right)
        'face_length': max_coords[2] - min_coords[2],  # Z axis (top-bottom)
        'face_depth': max_coords[1] - min_coords[1],   # Y axis (front-back)
        'center': (min_coords + max_coords) / 2
    }
    
    # Find bizygomatic breadth (maximum width at cheekbone level)
    mid_z = measurements['center'][2]
    face_height = measurements['face_length']
    
    # Filter vertices in cheekbone region (middle third of face)
    cheek_region = vertices[
        (vertices[:, 2] > mid_z - face_height/6) & 
        (vertices[:, 2] < mid_z + face_height/6)
    ]
    
    if len(cheek_region) > 0:
        measurements['bizygomatic_breadth'] = np.max(cheek_region[:, 0]) - np.min(cheek_region[:, 0])
    else:
        measurements['bizygomatic_breadth'] = measurements['face_width']
    
    # Menton-sellion length (chin to nose bridge)
    # Approximate: ~70% of total face length
    measurements['menton_sellion'] = measurements['face_length'] * 0.7
    
    return measurements

def process_all_headforms(headform_directory):
    """
    Process all STL headforms in a directory and create a reference database
    
    Expected filenames:
    - small_symmetry.stl
    - medium_symmetry.stl
    - large_symmetry.stl
    - long_narrow_symmetry.stl
    - short_wide_symmetry.stl
    """
    
    headform_dir = Path(headform_directory)
    
    # Expected headform files
    headform_files = {
        'small': 'small_symmetry.stl',
        'medium': 'medium_symmetry.stl',
        'large': 'large_symmetry.stl',
        'long_narrow': 'long_narrow_symmetry.stl',
        'short_wide': 'short_wide_symmetry.stl'
    }
    
    reference_data = {}
    
    for category, filename in headform_files.items():
        filepath = headform_dir / filename
        
        if filepath.exists():
            print(f"\nProcessing {category} headform...")
            vertices = read_stl_ascii(str(filepath))
            measurements = calculate_headform_measurements(vertices)
            
            reference_data[category] = {
                'bizygomatic_breadth': round(measurements['bizygomatic_breadth'], 1),
                'menton_sellion': round(measurements['menton_sellion'], 1),
                'face_width': round(measurements['face_width'], 1),
                'face_length': round(measurements['face_length'], 1),
                'face_depth': round(measurements['face_depth'], 1),
                'vertex_count': len(vertices)
            }
            
            print(f"  Bizygomatic Breadth: {measurements['bizygomatic_breadth']:.1f} mm")
            print(f"  Menton-Sellion: {measurements['menton_sellion']:.1f} mm")
            print(f"  Vertices: {len(vertices)}")
        else:
            print(f"Warning: {filename} not found")
    
    return reference_data

def create_reference_json(reference_data, output_file='headform_references.json'):
    """Save reference data to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(reference_data, f, indent=2)
    print(f"\nReference data saved to {output_file}")

def simplify_stl(input_file, output_file, target_vertex_count=10000):
    """
    Simplify an STL file by decimating vertices
    
    Note: This is a basic approach. For better results, use MeshLab or similar tools.
    """
    print(f"Simplifying {input_file}...")
    vertices = read_stl_ascii(input_file)
    
    # Simple decimation: keep every Nth vertex
    original_count = len(vertices)
    stride = max(1, original_count // target_vertex_count)
    
    simplified = vertices[::stride]
    
    # Write simplified STL
    with open(output_file, 'w') as f:
        f.write(f"solid simplified\n")
        
        # Write triangles (group vertices in sets of 3)
        for i in range(0, len(simplified) - 2, 3):
            v1, v2, v3 = simplified[i], simplified[i+1], simplified[i+2]
            
            # Calculate normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write(f"    outer loop\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
            f.write(f"    endloop\n")
            f.write(f"  endfacet\n")
        
        f.write(f"endsolid simplified\n")
    
    print(f"  Original: {original_count} vertices")
    print(f"  Simplified: {len(simplified)} vertices")
    print(f"  Saved to: {output_file}")

def main():
    """Main function - example usage"""
    
    print("=" * 60)
    print("NIOSH STL Headform Analysis Tool")
    print("=" * 60)
    
    # Example: Process a single headform
    print("\nAnalyzing large_symmetry.stl...")
    vertices = read_stl_ascii('large_symmetry.stl')
    measurements = calculate_headform_measurements(vertices)
    
    print(f"\nMeasurements:")
    print(f"  Bizygomatic Breadth: {measurements['bizygomatic_breadth']:.1f} mm")
    print(f"  Menton-Sellion Length: {measurements['menton_sellion']:.1f} mm")
    print(f"  Face Width: {measurements['face_width']:.1f} mm")
    print(f"  Face Length: {measurements['face_length']:.1f} mm")
    print(f"  Face Depth: {measurements['face_depth']:.1f} mm")
    print(f"  Total Vertices: {len(vertices):,}")
    
    # Example: Simplify the headform
    print("\n" + "=" * 60)
    print("Creating simplified version...")
    simplify_stl('large_symmetry.stl', 'large_symmetry_simplified.stl', target_vertex_count=10000)
    
    print("\n" + "=" * 60)
    print("\nTo process all headforms:")
    print("1. Place all 5 STL files in a directory:")
    print("   - small_symmetry.stl")
    print("   - medium_symmetry.stl")
    print("   - large_symmetry.stl")
    print("   - long_narrow_symmetry.stl")
    print("   - short_wide_symmetry.stl")
    print("\n2. Run: reference_data = process_all_headforms('path/to/directory')")
    print("3. Run: create_reference_json(reference_data)")
    print("\nThis will create headform_references.json for the app to use.")

if __name__ == "__main__":
    main()
