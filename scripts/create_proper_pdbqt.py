#!/usr/bin/env python3
"""
Create proper PDBQT format.
"""

def create_proper_pdbqt():
    """Create a proper PDBQT file format with correct spacing."""
    # PDBQT format with proper spacing for all fields
    content = """REMARK  VINA RESULT:    -6.5      0.000      0.000
ATOM      1  C   UNL     1      12.345   23.456   34.567  1.00  0.00     0.000 C
ATOM      2  N   UNL     1      13.345   24.456   35.567  1.00  0.00     0.000 N
ATOM      3  O   UNL     1      14.345   25.456   36.567  1.00  0.00     0.000 O
"""
    
    with open("test_proper.pdbqt", 'w') as f:
        f.write(content)
    
    # Test parsing
    with open("test_proper.pdbqt", 'r') as f:
        for i, line in enumerate(f):
            print(f"Line {i}: '{line.rstrip()}' (length: {len(line.rstrip())})")
            if line.startswith('ATOM'):
                # PDBQT format: columns 30-38 for X, 38-46 for Y, 46-54 for Z
                x_str = line[30:38].strip()
                y_str = line[38:46].strip()
                z_str = line[46:54].strip()
                atom_type = line[77:79].strip()
                
                print(f"  X: '{x_str}'")
                print(f"  Y: '{y_str}'")
                print(f"  Z: '{z_str}'")
                print(f"  Atom type: '{atom_type}'")
                
                try:
                    x = float(x_str)
                    y = float(y_str)
                    z = float(z_str)
                    print(f"  Parsed coordinates: ({x}, {y}, {z})")
                except ValueError as e:
                    print(f"  Error parsing coordinates: {e}")

if __name__ == "__main__":
    create_proper_pdbqt()
