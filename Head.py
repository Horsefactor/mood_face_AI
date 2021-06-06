from abc import ABC, abstractmethod
from typing import List
import HeadWatcher

class Head(ABC):
    """
    The Observer interface declares the update method, used by subjects.
    """
    colors = ['blue','red','green']





    @abstractmethod
    def update(self, headwatcher: HeadWatcher) -> None:
        """
        Receive update from subject.
        """
        pass

    @abstractmethod
    def update_pos(self, x,y,w,h) -> None:

        pass

class HeadV1(Head):

    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.size = w*h
        self.valid = False
        self.frames = 0
        print("head found")

    def update(self, headwatcher: HeadWatcher) -> None:
        if self.size < 1000:
            print("ConcreteObserverA: Reacted to the event")

    def update_pos(self, x,y,w,h) -> None:

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        pass






if __name__ == "__main__":
    # The client code.
    watcher= HeadWatcher.HeadWatcherV1()
    watcher.verify()