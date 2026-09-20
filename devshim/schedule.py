class CancelJob: pass


class Job:
    def __init__(self, interval=1, scheduler=None):
        self.interval = interval
        self.scheduler = scheduler
        self.job_func = None
    def do(self, fn, *a, **k):
        self.job_func = fn
        return self
    def __getattr__(self, item):
        return self


class Scheduler:
    def __init__(self):
        self.jobs = []
    def every(self, interval=1):
        j = Job(interval, self)
        self.jobs.append(j)
        return j
    def run_pending(self): pass
    def clear(self, *a, **k):
        self.jobs = []
