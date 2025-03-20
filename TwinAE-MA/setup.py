from setuptools import setup, find_packages

setup(
    name='Twin_Autoencoder_Structure',  # New package name
    version='0.1.0',                    # Starting version
    packages=find_packages(),           # Automatically find packages in the directory
    description='Package implementing the Twin Autoencoder Structure for manifold alignment',
    author='Adam G. Rustad',
    url='https://github.com/rustadadam/Twin_Autoencoder_Structure',  
    include_package_data=True,
    install_requires=[  # List project dependencies
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
)
