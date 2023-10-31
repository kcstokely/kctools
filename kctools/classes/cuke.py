import hashlib, os, pickle

class Cuke():
    
    '''
        This is a class that can preserve itself
          to disk, and read itself back as well.
    '''
    
    def __init__(self,
        name,
        abs_dir = '',         
        rel_dir = '',
        ext     = 'pkl',
        xhash   = False
    ):
        self.name = str(name)
        assert self.name
        cuke_dir = os.path.join(abs_dir, rel_dir)
        os.makedirs(cuke_dir, exist_ok = True)
        token = name if not xhash else hashlib.md5(name.encode('utf-8')).hexdigest()
        self._path = os.path.join(cuke_dir, f'{token}.{ext}')

    def __repr__(self):
        return self._name

    def read(self):
        with open(self._path, 'rb') as fp:
            self.__dict__.update(pickle.load(fp))
    
    def write(self):
        with open(self._path, 'wb') as fp:
            pickle.dump(self.__dict__.copy(), fp)
