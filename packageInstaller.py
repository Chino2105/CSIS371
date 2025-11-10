from os import popen, system

# File: packageInstaller.py
# Authors: Daniel Cater, Edin Quintana, Ryan Razzano, and Melvin Chino-Hernandez
# Version: 11/5/2024
# Description: This program reads a list of required Python packages from a text file
# and installs any that are not already present in the current environment.

def installPackages():
    with open('packagesUsed.txt', 'r') as file:
        content = file.read()

    packages = content.split()

    neededPackage = ""

    with popen("pip list") as stream:
        output = stream.read()
        for package in packages:
            if not (package in output):
                neededPackage += " " + package

    if (neededPackage != ""):
        system("pip install " + neededPackage)