import sys  # For interacting with system, especially for exception details
from src.logger import logging  # Import the configured logger from logger.py

# Function to generate detailed error message
def error_message_detail(error, error_detail: sys):
    """
    This function generates a detailed error message with filename, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()  # Get exception details (type, value, and traceback)
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name where the error occurred
    error_message = "Error occurred in Python script [{0}] at line number [{1}] with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)  # Format the error message
    )
    return error_message  # Return the formatted error message


# Custom exception class that will be raised with detailed error information
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with the provided error message and details.
        """
        super().__init__(error_message)  # Initialize base Exception class
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Detailed error

    def __str__(self):
        """
        This method is called when the exception is printed or logged. 
        It returns the detailed error message.
        """
        return self.error_message  # Return the detailed error message


# Generalized function for error logging and raising custom exceptions
def log_and_raise_exception(error_message, error_detail: sys):
    """
    This function logs the error message using logging and raises a CustomException.
    """
    logging.error(error_message)  # Log the error message using the imported logger
    raise CustomException(error_message, error_detail)  # Raise the custom exception with detailed error


# only to check if the exception is running and creating logs 
# if __name__ == "__main__": 
#     try:
#         a = 1 / 0  # This will raise a ZeroDivisionError
#     except Exception as e:
#         logging.info("Division by zero")  # Log the info message
#         raise CustomException(e, sys)  # Raise the custom exception with detailed error
