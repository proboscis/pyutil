from setuptools import setup,find_packages
setup(
    name="proboscuis_util",
    version="1.0",
    description="proboscis's utility library",
    author="Kento Masui",
    author_email="nameissoap@gmail.com",
    packages=find_packages(include="proboscis"),
    install_requires=[
        "pyprind",
        "click"
    ])