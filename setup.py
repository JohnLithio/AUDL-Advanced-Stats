"""This file controls how the pip-installable package is set up."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="audl_advanced_stats",
    version="0.0.3",
    author="John Lithio",
    description="For downloading and analyzing AUDL advanced stats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnLithio/AUDL-Advanced-Stats",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
