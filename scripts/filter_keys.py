import sys

keys_path = sys.argv[1]
test_set = sys.argv[2]
keys_filtered_path = keys_path.replace('.key', '.%s.key' % test_set)

with open(keys_path) as keys_f:
    with open(keys_filtered_path, 'w') as keys_filtered_f:
        for line in keys_f:
            if not line.startswith(test_set):
                continue

            line_filtered = line.lstrip('%s.' % test_set)
            keys_filtered_f.write(line_filtered)
