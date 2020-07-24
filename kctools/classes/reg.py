################################################

from abc import ABC, abstractmethod
from collections import defaultdict

################################################

class Register(object):
    
    '''
    A Register() is a class with a register:

      ._reg = {event: [obj1, obj2, obj3, ... ], ... }

      of objects which trigger on event.

      Basically, you are a Register if you might
        generate events.

      Registered objects should have method:
        .trigger(event, registrar)
        
      If it does not call .register() with events,
        it should also have method:
          .trigger_events()
    '''

    def __init__(self):
        self._reg  = defaultdict(list)
    
    def register(self, obj, events = None):
        try:
            for event in (events if events else obj.trigger_events()):
                self._reg[event].append(obj)
        except AttributeError:
            pass
                
    def deregister(self, obj, events = None):
        for event in (events if events else self._reg):
            try:
                self._reg[event].remove(obj)
            except KeyError:
                pass

    def check(self, events):
        for event in events:
            for obj in self._reg[event]:
                obj.trigger(event, self)

################################################

def Triggerable(ABC):
    
    @abstractmethod
    def trigger(self, event, registrar):
        pass

################################################




