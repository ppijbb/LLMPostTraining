
import os

target_file = '/home/conan/workspace/llm_training/models/spectra_model.py'
new_code_file = '/home/conan/workspace/llm_training/models/new_router_code.py'

# Read original file
with open(target_file, 'r') as f:
    lines = f.readlines()

# Read new code
with open(new_code_file, 'r') as f:
    new_code = f.read()

# Make sure new code ends with newline if not present, but usually read() gets all.
# We might want to ensure spacing.

# Define split points
start_idx = 1187 # Line 1188 (0-indexed)
end_idx = 2160   # Line 2161 (0-indexed)

# Verify context slightly to ensure we are cutting the right thing
print(f"Line at start_idx ({start_idx+1}): {lines[start_idx].strip()}")
print(f"Line at end_idx ({end_idx+1}): {lines[end_idx].strip()}")

expected_start = "class SPECTRARouter(nn.Module):"
expected_end = "iterations = 0"

if expected_start not in lines[start_idx]:
    print(f"WARNING: Expected '{expected_start}' but got '{lines[start_idx].strip()}'")
    # Don't abort, just warn? NO, abort if critical mismatch to avoid ruining file.
    # But whitespace might differ.
    if "class SPECTRARouter" not in lines[start_idx]:
         print("CRITICAL MISMATCH at start. Aborting.")
         exit(1)

if expected_end not in lines[end_idx]:
    print(f"WARNING: Expected '{expected_end}' but got '{lines[end_idx].strip()}'")
    # "iterations = 0" might be on line 2161.
    if "iterations" not in lines[end_idx]:
         print("CRITICAL MISMATCH at end. Aborting.")
         exit(1)

# Construct new content
new_lines = lines[:start_idx]
new_lines.append(new_code + "\n")
new_lines.extend(lines[end_idx:])

# Write back
with open(target_file, 'w') as f:
    f.writelines(new_lines)

print("Successfully replaced SPECTRARouter class.")
