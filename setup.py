# Base line file to Package the project like a python library.

from setuptools import find_packages, setup
from typing import List

hypen_e_dot = "-e ."
def get_requirements(file_path:str)->List[str]:
    """
    Reads a file containing a list of package requirements and returns a list of strings.
    
    Parameters
    ----------
    file_path : str
        The path to the file containing the list of package requirements
    
    Returns
    -------
    List[str]
        A list of strings containing the package requirements
    """
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        # Also remove hypen_e_dot
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
        
    return requirements

req = get_requirements("requirement.txt")

setup(
    name='Retail_Recommender',
    packages=find_packages(),
    version='1.0',
    description='Retail Items Recommednation system for ultility stores',
    author='Koelin',
    author_email='Koelinkrishh@gmail.com',
    install_requires = req
)