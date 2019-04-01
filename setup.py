##########################################################################################################################################################

import setuptools

with open('README.md', 'r') as fp:
    long_description = fp.read()
    
setuptools.setup(
     name         = 'kctools',  
     version      = '0.0.1',
     author       = 'kevin c. stokely',
     author_email = 'kcstokely@gmail.com',
     description  = 'miscellaneous elves',
     long_description = long_description,
     long_description_content_type = 'text/markdown',
     url          = 'https://github.com/kcstokely/kctools',
     packages     = setuptools.find_packages(),
     classifiers  = [
         'Operating System :: OS Independent',
         'Programming Language :: Python :: 3',
         'License :: OSI Approved :: MIT License'
     ]
)


