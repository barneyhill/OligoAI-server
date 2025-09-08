# rp_handler.py
import os
import runpod

# The path where the network volume is mounted
VOLUME_PATH = "/runpod-volume"

def handler(job):
    """
    Accesses the attached network volume directly to 'ls' the files
    and returns the list as a test output.
    """
    job_input = job.get('input', {})
    file_list = []
    target_directory = VOLUME_PATH

    # You can optionally specify a subdirectory in your input
    subdirectory = job_input.get('subdirectory')
    if subdirectory:
        # Basic security check to prevent directory traversal
        clean_subdir = os.path.normpath(subdirectory).lstrip('/')
        target_directory = os.path.join(VOLUME_PATH, clean_subdir)

    if not os.path.exists(target_directory) or not os.path.isdir(target_directory):
        return {"error": f"Directory not found in network volume: {target_directory}"}

    try:
        # This is how you can 'ls' the directory's contents
        file_list = os.listdir(target_directory)
    except Exception as e:
        return {"error": f"Failed to list files in '{target_directory}'. Error: {str(e)}"}

    # Generate a test output
    output = {
        "message": "Job completed successfully using direct file access!",
        "directory_inspected": target_directory,
        "files_found": file_list
    }

    return output

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
