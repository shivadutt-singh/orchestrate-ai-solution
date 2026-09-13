import os
import zipfile
import sys

def package_submission():
    print("Starting packaging process...")
    
    # Required deliverables
    output_file = "output.csv"
    chat_file_1 = "log.txt"
    chat_file_2 = "chat_transcript"
    usage_file = os.path.join("code", "evaluation", "usage_report.md")
    
    # 1. Verify files exist
    if not os.path.exists(output_file):
        print(f"ERROR: {output_file} not found at root.")
        sys.exit(1)
        
    chat_transcript = chat_file_1 if os.path.exists(chat_file_1) else chat_file_2
    if not os.path.exists(chat_transcript):
        print(f"ERROR: No chat transcript ({chat_file_1} or {chat_file_2}) found at root.")
        sys.exit(1)
        
    if not os.path.exists(usage_file):
        print(f"ERROR: {usage_file} not found.")
        sys.exit(1)
        
    # 2. Zip code/ directory
    zip_filename = "code.zip"
    exclude_dirs = {'__pycache__', '.git', 'venv'}
    exclude_files = {'.env', '.DS_Store'}
    
    print(f"Zipping code/ into {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("code"):
            # Exclude directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file not in exclude_files and not file.endswith('.pyc'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=".")
                    zipf.write(file_path, arcname)
                    
    print("\n--- Final Success Checklist ---")
    print(f" [x] {zip_filename} generated successfully.")
    print(f" [x] {output_file} verified.")
    print(f" [x] {chat_transcript} verified.")
    print("\nThe 3 required deliverables are ready for upload.")
    print("Submission URL: https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait/submission")

if __name__ == "__main__":
    package_submission()
