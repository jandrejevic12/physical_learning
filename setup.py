import setuptools

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

setuptools.setup(
    name="physical_learning",
    version="0.0.1",
    author="jandrejevic12",
    author_email="",
    description="A package physical learning with elastic networks",
    long_description="Physical learning with elastic networks",
    url="https://github.com/jandrejevic12/physical_learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy"]
)