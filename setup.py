import setuptools

setuptools.setup(
name="neva",
version="0.0.0",
author="Nathan Bouvier",
description="A software framework for NEVA application examples",
classifiers=[
"Programming Language :: Python :: 3",
"License :: MIT",
"Operating System :: OS Independent",
],
keywords=[
""
],
url="TODO",
packages=setuptools.find_packages(exclude=["graphs","lava_nc-0.9.0","example"]),
install_requires=[
"numpy",
"lava-nc",
"python-sat",
"matplotlib",
"networkx",
"tsplib95"
],
python_requires=">=3.8.0",
)
