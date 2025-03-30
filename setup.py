# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name="alltheplots",
    version="0.1.0",
    description="Intuitive automatic plotting for arrays of any dimensionality",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gomezzz/alltheplots",
    author="Pablo Gomez",
    author_email="contact@pablo-gomez.net",
    include_package_data=True,
    install_requires=[
        "loguru>=0.6.0",
        "matplotlib>=3.6.0",
        "numpy>=1.23.5",
        "scipy>=1.9.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.0.0",
        "umap-learn>=0.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    project_urls={
        "Source": "https://github.com/gomezzz/alltheplots/",
        "Bug Reports": "https://github.com/gomezzz/alltheplots/issues",
    },
)
