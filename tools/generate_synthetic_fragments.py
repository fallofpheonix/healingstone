import random
import argparse
from pathlib import Path

def generate_ply_file(output_path, fragment_id, num_points=50):
    """Generates a minimal valid ASCII PLY file using standard library only."""
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
    
    with open(output_file, "w") as f:
        f.write("\n".join(header) + "\n")
        for _ in range(num_points):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            
    print(f"Generated {output_file} ({num_points} points)")

def main():
    parser = argparse.ArgumentParser(description="Generate minimal synthetic 3D fragments (No dependencies).")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--num-fragments", type=int, default=10, help="Number of fragments to generate")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_fragments):
        generate_ply_file(args.output, i)

if __name__ == "__main__":
    main()
