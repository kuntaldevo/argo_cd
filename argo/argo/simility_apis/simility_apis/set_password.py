"""
Function for setting the password to the Simility instance for the API 
modules
"""
import getpass


def set_password() -> None:
    """Sets the password used for API calls to the platform"""

    global PASSWORD
    PASSWORD = getpass.getpass(
        'Please provide your password for logging into the Simility platform: ')
