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

    head_list = []
    valid_heads_counter = 0
    valid_head_list = []


    """
    List of subscribers. In real life, the list of subscribers can be stored
    more comprehensively (categorized by event type, etc.).
    """
    def __init__(self,positions=[],sensitivity = 150,timeFrame = 5):
        self.positions = positions
        self.sensitivity = sensitivity
        self.timeFrame = timeFrame

    def validate(self, head: Head) -> None:
        print("head validated")
        head.valid = True
        head.setColor(self.valid_heads_counter)
        self.valid_heads_counter += 1
        self.valid_head_list.append(head)


    def unvalidate(self, head: Head) -> None:
        print("Head unvalidated")
        self.valid_head_list.remove(head)

    def attach(self, head: Head) -> None:
        print("Something Detected")
        self.head_list.append(head)


    def detach(self, head: Head) -> None:
        print("Head removed")
        self.head_list.remove(head)

    """
    The subscription management methods.
    """

    def notify(self) -> None:
        """
        Trigger an update in each subscriber.
        """

        print("Subject: Notifying observers...")
        for head in self.head_list:
            head.update(self)

    def load_position(self,positions):
        self.positions = positions
        pass

    def verify(self) -> None:
        if len(self.head_list) < len(self.positions):
            for position in self.positions:
                newHead=True
                x, y, w, h = position
                for head in self.head_list:
                    if (x - head.x) ** 2 + (y - head.y) ** 2 < self.sensitivity ** 2:
                        newHead=False
                if newHead:
                    head = Head.HeadV1(x, y, w, h)
                    self.attach(head)


        else:
            for head in self.head_list: #check if existing heads are still valid
                if not head.valid:  #head not validates
                    stillhereInv= False
                    for position in self.positions:
                        x, y, w, h = position
                        if (x - head.x) ** 2 + (y - head.y) ** 2 < self.sensitivity ** 2:
                            head.update_pos(x, y, w, h)
                            head.frames += 1
                            stillhereInv= True
                            if head.frames >= self.timeFrame:
                                self.validate(head)

                    if not stillhereInv:
                        self.detach(head)
                else:           # if a head is supposed to be valid
                    stillhere =False
                    for position in self.positions:
                        x, y, w, h = position
                        if (x - head.x) ** 2 + (y - head.y) ** 2 < self.sensitivity**2:
                            head.update_pos(x, y, w, h)
                            stillhere=True
                            head.frames = self.timeFrame
                    if stillhere == False:
                        head.frames -= 1
                        if head.frames < 0:
                            head.valid = False
                            self.unvalidate(head)
                            self.detach(head)



