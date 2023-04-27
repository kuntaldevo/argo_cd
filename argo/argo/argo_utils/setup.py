# Authors: James Laidler <jlaidler@paypal.com>
# License: BSD 3 clause
import setuptools

setuptools.setup(
    name="argo_utils",
    version="1.0.0",
    author="Simility Data Team",
    packages=setuptools.find_packages(
        exclude=['examples']),  # + ['rule_optimisation']),
    # package_dir={
    #     'rule_optimisation': '../rule_optimisation/'},
    # dependency_links=["../rule_optimisation"],
    install_requires=['pytest==6.0.1', 'numpy==1.19.4', 'pandas==1.1.4',
                      'scikit-learn==0.23.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
