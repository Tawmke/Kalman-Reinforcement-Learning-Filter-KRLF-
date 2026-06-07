

def epoch2time_m(ep):
    """ calculate time from epoch """
    doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    year = int(ep[0:4])
    mon = int(ep[4:6])
    day = int(ep[6:8])
    hour = int(ep[8:10])
    min = int(ep[10:12])
    sec = float(ep[12:-1])/100

    if year < 1970 or year > 2099 or mon < 1 or mon > 12:
        return 'error'
    days = (year-1970)*365+(year-1969)//4+doy[mon-1]+day-2
    if year % 4 == 0 and mon >= 3:
        days += 1
    time = days*86400+hour*3600+min*60+sec
    # time.sec = ep[5]-sec
    return time

def time2epoch_m(t):
    """ convert time to epoch """
    mday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31,
            30, 31, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31,
            30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    days = int(t.time/86400)
    sec = int(t.time-days*86400)
    day = days % 1461
    for mon in range(48):
        if day >= mday[mon]:
            day -= mday[mon]
        else:
            break
    ep = [0, 0, 0, 0, 0, 0]
    ep[0] = 1970+days//1461*4+mon//12
    ep[1] = mon % 12+1
    ep[2] = day+1
    ep[3] = sec//3600
    ep[4] = sec % 3600//60
    ep[5] = sec % 60+t.sec
    return ep