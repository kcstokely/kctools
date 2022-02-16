################################################

from abc import ABC, abstractmethod
from collections import defaultdict

################################################

class Register():
    
    '''
        A Register() is a class with a register:

          ._reg = {event: [obj1, obj2, obj3, ... ], ... }

          of objects which trigger on an event.

          Basically, you are a Register if you might
            generate events.

          The registered objects should have method:
            .trigger(event, registrar)

          If it does not call .register() with events,
            it can also have method: .trigger_events()
    '''

    def __init__(self):
        self._reg  = defaultdict(list)
    
    def register(self, obj, events = None):
        if events is None:
            try:
                events = obj.trigger_events()
            except AttributeError:
                events = []
        for event in events:
            self._reg[event].append(obj)

    def deregister(self, obj, events = None):
        events = events if events is not None else self._reg
        for event in events:
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
    
    '''
        You are triggerable if events affect you.
    '''
    
    @abstractmethod
    def trigger(self, event, registrar):
        pass

################################################




