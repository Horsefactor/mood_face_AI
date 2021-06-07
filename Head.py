from abc import ABC, abstractmethod
from typing import List
import HeadWatcher

class Head(ABC):
    """
    The Observer interface declares the update method, used by subjects.
    """
    colors = [(223,32,51),(32,58,223),(0,198,7)]

    @abstractmethod
    def setColor(self,num) -> None:

        pass

    @abstractmethod
    def setEmotion(self, emo) -> None:
        pass


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
        self.emo = 4
        self.color = 'white'
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.size = w*h
        self.lastX=x
        self.lastY=y
        self.lastW=w
        self.lastH=h
        self.valid = False
        self.frames = 0
        print("head found")

    def update(self, headwatcher: HeadWatcher) -> None:
        if self.size < 1000:
            print("ConcreteObserverA: Reacted to the event")

    def update_pos(self, x,y,w,h) -> None:
        self.lastX = self.x
        self.lastY=self.y
        self.lastW=self.w
        self.lastH=self.h


        self.x = x
        self.y = y
        self.w = w
        self.h = h
        pass

    def setColor(self, num) -> None:

        self.color = Head.colors[num%len(Head.colors)]
        pass

    def setEmotion(self, emo) -> None:
        self.emo = emo







if __name__ == "__main__":
    # The client code.
    watcher= HeadWatcher.HeadWatcherV1()
    watcher.verify()