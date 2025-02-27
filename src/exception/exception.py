import sys
from src.logging.logger import logging

def get_error_message(error, error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message =f"Error occured in python script name {filename} line number {exc_tb.tb_lineno} and error message{str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = get_error_message(error_message, error_details=error_details)

    def __str__(self):
        return self.error_message
 

# if __name__=="__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("Division by zero error")
#         raise CustomException(e, sys)

