################################################

from abc import ABC, abstractmethod
from collections import defaultdict

################################################

class Register():
    
    '''
        A Register() is a class with an internal register:

          ._reg = {event: {obj1, obj2, obj3, ... }, ... }

          of objects which 'trigger' on occurance of an event.

          Basically, you need a Register if you may
            generate 'events' that other objects need
            to know about.
    
          These objects register with you, and if you ever
            generate the triggerable event, then you call
            their .trigger() method (with the event and
            yourself as arguments).
            
          These objects can also deregister from you when needed.
          
          Call the self.check() method on any generated events.
    '''

    def __init__(self):
        self._reg  = defaultdict(set)
    
    ###
    
    def register(self, obj, events = None):

        if events is None:
            try:
                events = obj.events()
            except AttributeError:
                pass
        
        if not isinstance(events, list):
            events = [ events ]
        
        for event in events:
            self._reg[event].add(obj)

    ###

    def deregister(self, obj, events = None):
        
        if events is not None:
            events = list(self._reg.keys())
        
        if not isinstance(events, list):
            events = [ events ]

        for event in events:
            try:
                self._reg[event].remove(obj)
            except KeyError:
                pass

    ###
                                                        
    def check(self, events):
    
        if not isinstance(events, list):
            events = [ events ]
               
        for event in events:
            for obj in self._reg[event]:
                try:
                    obj.trigger(event, self)
                except AttributeError:
                    pass   
                
################################################

def Triggerable(ABC):

    '''
        You are triggerable if events affect you.
        
        Register yourself with an event-generating Register(),
          with a list of events you are interested in. If ever
          the Register() generates a corresponding event, your
          .trigger() method is called.
          
        If you do not call Register.register() with an explicit
          a list of events, you can have .events() method which
          returns the list of events.
          
        Otherwise, registering will have no effect.
    '''

    @abstractmethod
    def trigger(self, event, registrar):
        pass

    @abstractmethod
    def events(self):
        return []

################################################




