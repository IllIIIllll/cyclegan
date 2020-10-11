import datetime
import timeit
import time

class Timer:
    def __init__(self, fmt='s', print_at_exit=True, timer=timeit.default_timer):
        assert fmt in ['ms', 's', 'datetime'], "`fmt` should be 'ms', 's' or 'datetime'!"
        self._fmt = fmt
        self._print_at_exit = print_at_exit
        self._timer = timer
        self.start()

    def __enter__(self):
        self.restart()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._print_at_exit:
            print(str(self))

    def __str__(self):
        return self.fmt(self.elapsed)[1]

    def start(self):
        self.start_time = self._timer()

    restart = start

    @property
    def elapsed(self):
        return self._timer() - self.start_time

    def fmt(self, second):
        if self._fmt == 'ms':
            time_fmt = second * 1000
            time_str = f'{time_fmt} {self._fmt}'
        elif self._fmt == 's':
            time_fmt = second
            time_str = f'{time_fmt} {self._fmt}'
        elif self._fmt == 'datetime':
            time_fmt = datetime.timedelta(seconds=second)
            time_str = str(time_fmt)
        return time_fmt, time_str


def timeit(run_times=1, **timer_kwargs):
    def decorator(f):
        def wrapper(*args, **kwargs):
            timer_kwargs.update(print_at_exit=False)
            with Timer(**timer_kwargs) as t:
                for _ in range(run_times):
                    out = f(*args, **kwargs)
            print(f'[*] Execution time of function "{f.__name__}" for {run_times} runs is {t} = {t.fmt(t.elapsed / run_times)[1]} * {run_times} [*]')
            return out
        return wrapper

    return decorator


if __name__ == "__main__":
    print(1)
    with Timer() as t:
        time.sleep(1)
        print(t)
        time.sleep(1)

    with Timer(fmt='datetime') as t:
        time.sleep(1)

    print(2)
    t = Timer(fmt='ms')
    time.sleep(2)
    print(t)

    t = Timer(fmt='datetime')
    time.sleep(1)
    print(t)

    print(3)

    @timeit(run_times=5, fmt='s')
    def blah():
        time.sleep(2)

    blah()
