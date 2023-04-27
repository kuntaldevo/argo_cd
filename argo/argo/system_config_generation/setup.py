# Authors: James Laidler <jlaidler@paypal.com>
# License: BSD 3 clause
import setuptools
setuptools.setup(
    name="system_config_generation",
    version="1.0.0",
    author="Simility Data Team",
    packages=setuptools.find_packages(exclude=['examples']),
    install_requires=['pytest==6.0.1', 'numpy==1.19.4', 'pandas==1.1.4'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
