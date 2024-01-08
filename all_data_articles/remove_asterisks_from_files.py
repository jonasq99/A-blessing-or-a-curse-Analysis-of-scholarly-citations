import os


def rename_files(directory):
    for filename in os.listdir(directory):
        new_filename = filename.replace('*', '')

        # remove all special character except - and _
        new_filename = ''.join(e for e in new_filename if e.isalnum() or e == '-' or e == '_' or e == '.')

        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)

        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_filename}'")


directory_path = "."
rename_files(directory_path)
