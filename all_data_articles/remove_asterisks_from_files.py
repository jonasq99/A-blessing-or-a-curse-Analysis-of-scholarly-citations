import os


def rename_files(directory):
    for filename in os.listdir(directory):
        if '*' in filename:
            new_filename = filename.replace('*', '')

            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)

            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_filename}'")


directory_path = "."
rename_files(directory_path)
