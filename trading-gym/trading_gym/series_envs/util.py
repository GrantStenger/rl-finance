from datetime import datetime

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DAILY_TIME_FORMAT = '%Y-%m-%d'

def mk_time(s, daily=False):
    if daily:
        return datetime.strptime(s, DAILY_TIME_FORMAT)
    else:
        return datetime.strptime(s, TIME_FORMAT)


def mk_str(s, daily=False):
    if daily:
        return s.strftime(DAILY_TIME_FORMAT)
    else:
        return s.strftime(TIME_FORMAT)


"""
Convert a string from format 'HH:MM' to datetime
"""
def mk_time_hours(x):
    x_parts = x.split(':')
    return float(x_parts[0]) + float(x_parts[1])/60.0


"""
This will generate a list of length N.
Including the end_dt
"""
def minute_range_before(end_dt, N):
    i = N
    while i >= 0:
        yield end_dt - datetime.timedelta(minutes=x)
        i += 1


def sub_mins(dt, mins):
    return dt - datetime.timedelta(minutes=mins)



def sub_days(dt, days):
    return dt - datetime.timedelta(days=days)
