from setuptools import find_packages, setup

__version__ = "1.0.0"

setup(
    name="fast_ivf",
    version=__version__,
    description="Efficient implementation of IVF + IVFPQ Index with numpy and numba",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/kmkolasinski/fast-ivf",
    author="Krzysztof Kolasinski",
    author_email="kmkolasinski@gmail.com",
    packages=find_packages(exclude=["tests", "notebooks"]),
    include_package_data=False,
    zip_safe=False,
    install_requires=[
        "numba>=0.58.0",
        "numpy>=1.24.3",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
