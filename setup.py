import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    # project details
    name="cyclum-mokab",
    version="0.1.0",
    description="Inferring pseudotime in scRNA-seq data using an autoencoder with the Fourier basis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KChen-lab/Cyclum",
    license='MIT',
    
    # author details
    author="Masaaki Okabe",
    author_email="mokab.0328@gmail.com",

    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy', 'pandas', 'scikit-learn', 'torch', 'jupyter', 'matplotlib'] 
)
