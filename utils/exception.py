class ModelExecuteError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def do_something(value):
    if value < 0:
        raise MyCustomError("Negative value is not allowed")
    return value

try:
    result = do_something(-1)
except MyCustomError as e:
    print(e)