################################################

import hashlib, os, pickle

'''
    This is a class that can preserve itself
      to disk, and load itself back as well.
'''

class Cuke():
    
    def __init__(self, name,
        c_dir = os.path.join(os.getcwd(), 'cukes'),
        ext   = 'pkl',
        xhash = False,
        force = False
    ):
        
        assert name
        self.name = str(name)
        self._dir = c_dir
        if not os.path.exists(c_dir):
            os.makedirs(c_dir)
        self._name = name if not xhash else hashlib.md5(name.encode('utf-8')).hexdigest()
        self._path = os.path.join(c_dir, f'{self._name}.{ext}')
        if os.path.exists(self._path) and not force:
            raise Exception('Existe déjà.')

    def __repr__(self):
        return self.name

    def read(self, state):
        with open(self._path, 'rb') as fp:
            self.__dict__.update(pickle.load(fp))
    
    def write(self):
        with open(self._path, 'wb') as fp:
            pickle.dump(self.__dict__.copy(), fp)

################################################






