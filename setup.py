from setuptools import setup, find_packages

setup(
    name='ArduPilotLogReviewer',
    version='0.1.0',
    description='A Python-based tool for automated analysis and visualisation of ArduPilot DataFlash logs.',
    author="Sukhbir Mahal",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "pymavlog",
        "scipy",
    ],
    python_requires='>=3.13',
)