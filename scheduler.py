import time

class StepScheduler:
    def __init__(self):
        self._next = {}

    def due(self, key: str, interval_s: float, now: float) -> bool:
        nxt = self._next.get(key, -1e9)
        if now >= nxt:
            self._next[key] = now + interval_s
            return True
        return False