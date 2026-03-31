import random
import argparse
import math
from pathlib import Path

def generate_alignable_ply(output_path, fragment_id, num_points=200):
    """
    Generates a PLY file representing a face of a cube.
    All fragments will occupy the same region (z=0, x/y in [-1, 1]) 
    so that identity alignment (Identity Matrix) works.
    """
    output_file = Path(output_path) / f"fragment_{fragment_id:03}.ply"
    
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]
    
    # Use a fixed seed for reproducibility within each fragment
    rand = random.Random(fragment_id)
    
    with open(output_file, "w") as f:
        f.write("\n".join(header) + "\n")
        for _ in range(num_points):
            # Generate points on a flat plane with slight noise
            x = rand.uniform(-1, 1)
            y = rand.uniform(-1, 1)
            z = rand.uniform(-0.01, 0.01) # Slightly non-planar but very alignable
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            
    print(f"Generated alignable {output_file} ({num_points} points)")

def main():
    parser = argparse.ArgumentParser(description="Generate alignable synthetic 3D fragments (No dependencies).")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--num-fragments", type=int, default=5, help="Number of fragments to generate")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_fragments):
        generate_alignable_ply(args.output, i)

if __name__ == "__main__":
    main()
