# Authors: James Laidler <jlaidler@paypal.com>
# License: BSD 3 clause
import setuptools
setuptools.setup(
    name="argo",
    version="2.0.0",
    author="Simility Data Team",
    description="Rule generation, optimisation and scoring library",
    packages=setuptools.find_packages(exclude=['examples']),
    install_requires=['category-encoders==2.0.0', 'matplotlib==3.0.3',
                      'seaborn==0.9.0', 'numpy==1.19.4', 'pandas==1.1.4',
                      'hyperopt==0.2.5', 'joblib==0.16.0', 'pytest==6.0.1',
                      'requests==2.20.1', 'httpretty==1.0.5',
                      'scikit-learn==0.23.2', ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
