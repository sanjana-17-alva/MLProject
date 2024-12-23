# import logging  # Import the logging module to handle logging messages
# import os  
# from datetime import datetime  

# # Create a log file name based on the current date and time in the format MM_DD_YYYY_HH_MM_SS.log
# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# # Define the path for the logs folder and create the path for the log file
# logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# # Create the 'logs' directory if it doesn't already exist (exist_ok=True prevents errors if the directory exists)
# os.makedirs(logs_path, exist_ok=True)

# # Define the full path for the log file, combining the logs directory path with the log file name
# LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# # Configure logging: setting the log file path, log format, and log level (INFO level and above will be logged)
# logging.basicConfig(
#     filename=LOG_FILE_PATH,  
#     format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define the format for log messages
#     level=logging.INFO  # Set the logging level to INFO (logs messages with level INFO and above)
# )

# # only to check if the logger code is running 
# # if __name__ == "__main__":
# #     logging.info("Logging has started")


import logging
import os
from datetime import datetime

# Create a log file name based on the current date and time in the format MM_DD_YYYY_HH_MM_SS.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path for the logs folder
logs_directory = os.path.join(os.getcwd(), "logs")

# Create the 'logs' directory if it doesn't already exist
os.makedirs(logs_directory, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(logs_directory, LOG_FILE)

# Configure logging: setting the log file path, log format, and log level (INFO level and above will be logged)
logging.basicConfig(
    filename=LOG_FILE_PATH,  
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define the format for log messages
    level=logging.INFO  # Set the logging level to INFO (logs messages with level INFO and above)
)

# Test the logging setup
# logging.info("Logger setup complete. This is a test log message.")


