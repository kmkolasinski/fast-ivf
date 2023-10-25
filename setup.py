from setuptools import find_packages, setup

from fast_ivf.version import __version__

setup(
    name="fast_ivf",
    version=__version__,
    description="",
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
)
