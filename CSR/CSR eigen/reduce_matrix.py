import sys

def reduce_matrix(input_file, output_file, new_size):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    header_index = 0
    while lines[header_index].startswith('%'):
        header_index += 1
    header = lines[header_index].strip().split()
    M, N, nnz = map(int, header)
    new_entries = []
    for line in lines[header_index+1:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        i, j, val = parts
        i = int(i)
        j = int(j)
        if i <= new_size and j <= new_size:
            new_entries.append((i, j, val))
    new_nnz = len(new_entries)
    with open(output_file, 'w') as f:
        f.write(f"{new_size} {new_size} {new_nnz}\n")
        for entry in new_entries:
            f.write(f"{entry[0]} {entry[1]} {entry[2]}\n")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: reduce_matrix.py input_file output_file new_size")
        sys.exit(1)
    reduce_matrix(sys.argv[1], sys.argv[2], int(sys.argv[3]))

