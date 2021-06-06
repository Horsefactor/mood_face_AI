from abc import ABC, abstractmethod
from typing import List
import Head

class HeadWatcher(ABC):
    """
    The Subject interface declares a set of methods for managing subscribers.
    """

    @abstractmethod
    def attach(self, head: Head) -> None:
        """
        Attach an observer to the subject.
        """
        pass

    @abstractmethod
    def detach(self, head: Head) -> None:
        """
        Detach an observer from the subject.
        """
        pass

    @abstractmethod
    def notify(self) -> None:
        """
        Notify all observers about an event.
        """
        pass

    def verify(self) -> None:
        pass


class HeadWatcherV1(HeadWatcher):
    """
    The Subject owns some important state and notifies observers when the state
    changes.
    """

    _state: int = None
    """
    For the sake of simplicity, the Subject's state, essential to all
    subscribers, is stored in this variable.
    """

    _head_list = []
    """
    List of subscribers. In real life, the list of subscribers can be stored
    more comprehensively (categorized by event type, etc.).
    """
    def __init__(self,positions=[]):
        self.positions = positions


    def attach(self, head: Head) -> None:
        print("Subject: Attached an observer.")
        self._head_list.append(head)

    def detach(self, head: Head) -> None:
        self._head_list.remove(head)

    """
    The subscription management methods.
    """

    def notify(self) -> None:
        """
        Trigger an update in each subscriber.
        """

        print("Subject: Notifying observers...")
        for head in self._head_list:
            head.update(self)

    def load_position(self,positions):
        self.positions = positions
        pass

    def verify(self) -> None:
        for position in self.positions:
            x,y,w,h = position
            if w + h >300:
                print("probably head")