import matplotlib.pyplot as plt

class DataPlotter:
    def __init__(self):
        self._data = {}

    def add(self, name, value):
        if name in self._data:
            self._data[name].append(value)
        else:
            self._data[name] = [value]

    def get(self, name):
        return self._data[name]

    def plot(self, name, figsize=(20,10), title=""):
        plt.figure(figsize=figsize)
        plt.plot(self.get(name))
        plt.title(title)
        plt.show()
