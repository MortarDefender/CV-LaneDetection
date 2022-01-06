import sys
from setuptools import setup


def getRequirements():
    with open("requirements.txt", "r") as f:
        read = f.read()

    return read.split("\n")


setup(
    name = 'lane detection',
    version= "1.0.1",
    description='lane detection for python',
    long_description='lane detection for python',
    author='Mortar Defender',
    license='MIT License',
    url = '__',
    setup_requires = getRequirements(),
    install_requires = getRequirements(),
    include_package_data=True
)
