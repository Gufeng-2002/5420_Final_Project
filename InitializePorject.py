# Initialize all essential directory and files for a project.
# Majorly include 4 parts: data, doc, results and src.
# In the doc directory, a simply organized latex frame is created for convenience.
# This latex frame comes a local path.
# ! This code should only run one time at the beginning of the project.

import os
import shutil
import argparse



def create_project_folders(base_path):
    """
    Creates the folders: doc, data, results, src inside the given directory.

    Args:
        base_path (str): The path where the folders should be created.
    """
    folders = ["doc", "data", "results", "src"]
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created: {folder_path}")


def copy_all_to_doc(source_folder, destination_folder):
    """
    Copies all files, folders, and sub-files from a source folder to the 'doc' folder.

    Args:
        source_folder (str): The folder containing the files and subfolders to copy.
        destination_folder (str): The 'doc' folder where files will be pasted.
    """
    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isdir(source_path):  # If it's a folder, copy it recursively
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:  # If it's a file, just copy it
            shutil.copy2(source_path, destination_path)

        print(f"Copied: {source_path} → {destination_path}")


# Example usage
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Initialize project directories and copy LaTeX files.")

    # Add command-line arguments
    parser.add_argument("base_directory", type=str, help="Base directory where project folders will be created.")
    parser.add_argument("destination_directory", type=str, help="Destination directory where files will be copied (usually 'doc').")

    # Parse arguments from the command line
    args = parser.parse_args()

    # Create project folders
    create_project_folders(args.base_directory)
    
    source_directory = "/Users/gufeng/code/latex_writing/report_format" 

    # Copy files from the source directory to the doc directory
    copy_all_to_doc(source_directory, args.destination_directory)