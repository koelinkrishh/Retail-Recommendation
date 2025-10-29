# For exception handling
import sys


def error_details(error, error_detail:sys):
    """ Function to get detailed error message with file name and line number.

    Parameters:
        error (Exception): The exception object containing the error message.
        error_detail (sys): The sys module containing the exception details.

    Returns:
        str: A detailed error message with file name and line number.
    """
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"""Error occured in [{file_name}], line number: [{exc_tb.tb_lineno}].
        => error message: [{str(error)}]"""
        
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_msg_detail:sys):
        # Inherit from exception class
        """ Constructor for CustomException class.

        Parameters:
        error_message (str): Message to be displayed when exception occurs
        error_msg_detail (sys): Details of the error for debugging purposes

        Returns:    None
        """
        super().__init__(error_message) # get base error class message
        self.error_msg = error_details(error_message, error_detail=error_msg_detail)
    
    def __str__(self):
        """ Return the error message string representation of the exception. """
        return self.error_msg

