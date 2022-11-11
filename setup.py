import setuptools

with open('README.md', 'r') as fp:
    long_description = fp.read()
    long_description_content_type = 'text/markdown'

setuptools.setup(
    name             = 'kctools',
    version          = '0.2.1',
    author           = 'kevin c. stokely',
    author_email     = 'kcstokely@gmail.com',
    description      = 'miscellaneous elves',
    url              = 'https://github.com/kcstokely/kctools',
    packages         = setuptools.find_packages(),
    install_requires = [
        'matplotlib',
        'numpy',
        'pandas',
        'scipy'
    ],
    classifiers = [
         'Operating System :: OS Independent',
         'Programming Language :: Python :: 3',
         'License :: OSI Approved :: MIT License'
    ],
    long_description              = long_description,
    long_description_content_type = long_description_content_type
)