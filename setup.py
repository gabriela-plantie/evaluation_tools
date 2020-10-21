import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evaluation_tools",
    version="0.0.2",
    author="Gabriela Plantie",
    author_email="glplantie@gmail.com",
    description="Evaluation Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabriela-plantie/evaluation_tools",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.1.2',
        'statsmodels>=0.12.0',
        'scipy>=1.5.2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)