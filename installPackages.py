import os
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