import os

# Splits a large JSONL file into smaller chunks based on file size.
def split_jsonl(input_file_path, max_size_gb=1):
    
    # Calculate the max size in bytes
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024
    
    print(f"Setting max chunk size to {max_size_gb} GB ({max_size_bytes} bytes).")

    try:
        # Open the large input file for reading
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            
            file_number = 1
            current_size = 0
            outfile = None

            output_directory = "./CORPUS/"
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)

            # Get the next output file
            def get_output_file(num):
                filename = f"miniCorpus_{num}.jsonl"
                filepath = os.path.join(output_directory, filename)
                print(f"Creating new file chunk: {filename}")
                return open(filepath, 'w', encoding='utf-8')

            # Start with the first output file
            outfile = get_output_file(file_number)

            try:
                # Read the input file line by line
                for line in infile:
                    # Get the size of the line in bytes
                    line_bytes = len(line.encode('utf-8'))
                    
                    # Check if adding this line would exceed the max size.
                    # We also check 'current_size > 0' to ensure that if a single
                    # line is larger than max_size_gb, it still gets written
                    # to its own file.
                    if current_size + line_bytes > max_size_bytes and current_size > 0:
                        # Close the current file
                        outfile.close()
                        
                        # Increment the file number and open a new file
                        file_number += 1
                        outfile = get_output_file(file_number)
                        
                        # Reset the current size for the new file
                        current_size = 0
                        
                    # Write the line to the current output file
                    outfile.write(line)
                    # Update the current file's size
                    current_size += line_bytes
            
            finally:
                # Ensure the last output file is closed
                if outfile and not outfile.closed:
                    outfile.close()
                    
        print(f"\nSuccessfully split '{input_file_path}' into {file_number} part(s).")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
        print("Please check the 'INPUT_FILE' variable in the script.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main execution ---
if __name__ == "__main__":
    
    # CURPUS FILE.
    CORPUSFILE = "trec-tot-2025-corpus.jsonl" 
    
    # Desired GB
    SPLIT_SIZE_GB = 0.5

    print(f"Starting to split '{CORPUSFILE}' into {SPLIT_SIZE_GB}GB chunks...")
    
    # Run the splitting function
    split_jsonl(CORPUSFILE, SPLIT_SIZE_GB)