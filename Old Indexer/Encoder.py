import os
import glob
import shutil
import time

# CONFIGURATION
input_corpus_dir = "../CORPUS"
temp_output_dir = "../indexes/temp_vectors"
final_output_dir = "../indexes/project_vectors"

# 1. Setup Directories
if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)

files = sorted([f for f in glob.glob(f"{input_corpus_dir}/*") if os.path.isfile(f)])
print(f"Found {len(files)} files to process.")

start_time = time.time()

# 2. Process One by One (Optimized for Speed)
for i, file_path in enumerate(files):
    print(f"\n--- Processing File {i + 1}/{len(files)}: {file_path} ---")

    shard_output = os.path.join(temp_output_dir, f"part_{i:03d}")

    # THE SPEED UPGRADE:
    # --batch 32: Processes 4x more data at once
    # --device mps: Uses your Mac's GPU instead of CPU
    # --fp16: Cuts memory usage in half so batch 32 fits
    cmd = (
        f"python -m pyserini.encode "
        f"input --corpus {file_path} --fields text "
        f"output --embeddings {shard_output} "
        f"encoder --encoder sentence-transformers/all-MiniLM-L6-v2 --batch 32 --device mps --fp16"
    )

    exit_code = os.system(cmd)

    if exit_code != 0:
        print(f"!!! CRASHED on file {file_path}. Stopping.")
        exit(1)

    # Move files
    generated_files = glob.glob(f"{shard_output}/*.jsonl")
    for gf in generated_files:
        new_name = f"embeddings_part_{i:03d}.jsonl"
        shutil.move(gf, os.path.join(final_output_dir, new_name))

    shutil.rmtree(shard_output)

total_time = (time.time() - start_time) / 60
print(f"\n✅ DONE! Processed {len(files)} files in {total_time:.1f} minutes.")