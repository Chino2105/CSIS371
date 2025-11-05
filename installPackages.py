import os

# File: installPackages.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/5/2024
# Description: This program reads a list of required Python packages from a text file
# and installs any that are not already present in the current environment.

with open('packagesUsed.txt', 'r') as file:
    content = file.read()

packages = content.split()

neededPackage = ""

with os.popen("pip list") as stream:
    output = stream.read()
    for package in packages:
        if not (package in output):
            neededPackage += " " + package

if (neededPackage != ""):
    os.system("pip install " + neededPackage)