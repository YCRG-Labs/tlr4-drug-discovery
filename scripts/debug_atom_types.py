#!/usr/bin/env python3
"""
Debug atom type parsing.
"""

def debug_atom_types():
    """Debug atom type parsing in PDBQT files."""
    sample_content = """REMARK  VINA RESULT:    -6.5      0.000      0.000
ATOM      1  C   UNL     1      12.345   23.456   34.567  1.00  0.00     0.000 C
ATOM      2  N   UNL     1      13.345   24.456   35.567  1.00  0.00     0.000 N
ATOM      3  O   UNL     1      14.345   25.456   36.567  1.00  0.00     0.000 O
"""
    
    with open("debug.pdbqt", 'w') as f:
        f.write(sample_content)
    
    # Test parsing
    with open("debug.pdbqt", 'r') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            print(f"Line {i}: '{line}' (length: {len(line)})")
            if line.startswith('ATOM'):
                # Check different column positions
                print(f"  Character 12-16 (atom name): '{line[12:16]}'")
                print(f"  Character 77-79 (atom type): '{line[77:79] if len(line) > 78 else 'N/A'}'")
                print(f"  Character 77-end: '{line[77:] if len(line) > 76 else 'N/A'}'")
                
                # Try parsing atom type
                atom_type = None
                if len(line) > 78:
                    atom_type = line[77:79].strip()
                elif len(line) > 76:
                    atom_type = line[77:].strip()
                else:
                    # Fall back to parsing from the atom name (column 12-16)
                    atom_name = line[12:16].strip()
                    atom_type = atom_name[0] if atom_name else 'C'
                
                print(f"  Parsed atom type: '{atom_type}'")
                
                # Test atom type conversion
                atom_type_map = {
                    'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
                    'H': 1, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53
                }
                atomic_num = atom_type_map.get(atom_type, 6)
                print(f"  Atomic number: {atomic_num}")

if __name__ == "__main__":
    debug_atom_types()
