#!/usr/bin/env python3
"""
Create final correct PDBQT format.
"""

def create_final_pdbqt():
    """Create a final correct PDBQT file format."""
    # PDBQT format with proper spacing - need to ensure Z coordinate has enough space
    content = """REMARK  VINA RESULT:    -6.5      0.000      0.000
ATOM      1  C   UNL     1      12.345   23.456   34.567  1.00  0.00     0.000 C
ATOM      2  N   UNL     1      13.345   24.456   35.567  1.00  0.00     0.000 N
ATOM      3  O   UNL     1      14.345   25.456   36.567  1.00  0.00     0.000 O
"""
    
    with open("test_final.pdbqt", 'w') as f:
        f.write(content)
    
    # Test parsing with the actual column positions
    with open("test_final.pdbqt", 'r') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            print(f"Line {i}: '{line}' (length: {len(line)})")
            if line.startswith('ATOM'):
                # Let's check the actual character positions
                print(f"  Character 30-38: '{line[30:38]}'")
                print(f"  Character 38-46: '{line[38:46]}'")
                print(f"  Character 46-54: '{line[46:54]}'")
                print(f"  Character 77-79: '{line[77:79] if len(line) > 78 else 'N/A'}'")
                
                # Try different column positions
                print(f"  X (30-38): '{line[30:38].strip()}'")
                print(f"  Y (38-46): '{line[38:46].strip()}'")
                print(f"  Z (46-54): '{line[46:54].strip()}'")
                
                # Try parsing with different positions
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    print(f"  SUCCESS: ({x}, {y}, {z})")
                except ValueError as e:
                    print(f"  FAILED: {e}")
                    
                    # Try alternative parsing
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            x = float(parts[5])
                            y = float(parts[6])
                            z = float(parts[7])
                            print(f"  ALTERNATIVE SUCCESS: ({x}, {y}, {z})")
                        except (ValueError, IndexError) as e2:
                            print(f"  ALTERNATIVE FAILED: {e2}")

if __name__ == "__main__":
    create_final_pdbqt()
