import os
import shutil
import datetime
from datetime import timedelta
import time
import pytz
import json
from multiprocessing.managers import DictProxy

def remove_expired_folders(directory, expiry_days=2):
    if not os.path.exists(directory):
        return
    print(f'# remove_expired_folders - Checking directory "{directory}" to remove files.')
    # Get the current date
    current_date = datetime.datetime.now()

    # List all folders in the specified directory
    for folder_name in os.listdir(directory):
        # Ensure it's a folder and follows the {yyyymmdd} format
        if os.path.isdir(os.path.join(directory, folder_name)) and len(folder_name) == 8:
            try:
                # Extract the date from the folder name
                folder_date = datetime.datetime.strptime(folder_name, '%Y%m%d')

                # Check if the folder is older than the expiry days
                if current_date - folder_date > timedelta(days=expiry_days):
                    # Remove the folder
                    folder_path = os.path.join(directory, folder_name)
                    shutil.rmtree(folder_path)
                    print(f"Removed expired folder: {folder_path}")

            except ValueError:
                # If the folder name is not in the correct format, ignore it
                continue
    print(f'# remove_expired_folders - Working done.')

def cleanup_loop(directory, interval_sec=1, expiry_days=2):
    while True:
        remove_expired_folders(directory, expiry_days)
        # Sleep for 24 hours (86400 seconds)
        # time.sleep(86400)
        time.sleep(interval_sec)
        

def save_json_file(file_path, json_msg):
    try:        
        directory = os.path.dirname(file_path)
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
                    
        dict_msg = {}
        if isinstance(json_msg, DictProxy):
            dict_msg = dict(json_msg)
        else:
            dict_msg = json_msg
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(dict_msg, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(e)
        
def load_json_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        return data
    except Exception as e:
        print(e)
        return None
    
def save_statistics(parent_dir, file_name, json_msg, create_sub_dir=True):
    try:
        if create_sub_dir:
            timezone_str = 'Asia/Seoul'
            timezone = pytz.timezone(timezone_str)  # Change to your desired timezone
            current_time = datetime.datetime.now(timezone)
            # formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
            directory_name = os.path.join(parent_dir, current_time.strftime('%Y%m%d'))
        else:
            directory_name = parent_dir
        file_path = os.path.join(directory_name, file_name)
        save_json_file(file_path, json_msg)
            
    except Exception as e:
        print(e)
        
def load_statistics(parent_dir, file_name, create_sub_dir=True):
    try:
        if create_sub_dir:
            timezone_str = 'Asia/Seoul'
            timezone = pytz.timezone(timezone_str)  # Change to your desired timezone
            current_time = datetime.datetime.now(timezone)
            directory_name = os.path.join(parent_dir, current_time.strftime('%Y%m%d'))
        else:
            directory_name = parent_dir
        file_path = os.path.join(directory_name, file_name)
        return load_json_file(file_path)
            
    except Exception as e:
        print(e)
        return None
        
def save_json_loop(parent_dir, file_name, json_msg, interval_sec=10000, create_sub_dir=True):
    while True:
        save_statistics(parent_dir, file_name, json_msg, create_sub_dir)
        time.sleep(interval_sec)

if __name__ == "__main__":
    # Example usage
    directory_path = '/path/to/your/directory'
    remove_expired_folders(directory_path)
