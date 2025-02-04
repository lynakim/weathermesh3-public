from datetime import datetime, timedelta

#os.environ['WB_CLIENT_ID'] = 'wm_da'
#os.environ['WB_API_KEY'] = 'wb_bced4575ca173ba69948a56beb0ef9b0'

start_year = 2016
end_year = 2025 # inclusive

start_date = datetime(start_year, 1, 1)
end_date = datetime(end_year + 1, 1, 1)

def reset_year(date):
    if date.month == 12:
        return date.replace(year=date.year+1, month=1)
    else:
        return date.replace(month=date.month+1)

while start_date < end_date:
    intermediate_date = reset_year(start_date)
    #print(f'windborne super-observations \'{start_date}\' \'{intermediate_date}\' csv -d /huge/proc/windborne/{start_date.year}{start_date.month:02d}')
    print(f'python3 norm/da/procwindborne.py {start_date.year}-{start_date.month:02d}-{start_date.day:02d}')
    
    start_date = reset_year(start_date)

# python3 norm/da/makewindbornecmds.py | parallel --ungroup --jobs 1