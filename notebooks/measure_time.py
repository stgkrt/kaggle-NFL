import os
import json
f = open("./kaggle.json", 'r')#kaggle.jsonが保存してあるパスを指定する。
json_data = json.load(f) 
os.environ['KAGGLE_USERNAME'] = json_data['username']
os.environ['KAGGLE_KEY'] = json_data['key']

from kaggle.api.kaggle_api_extended import KaggleApi
import datetime
from datetime import timezone
import time

api = KaggleApi()
api.authenticate()

COMPETITION =  'nfl-player-contact-detection'
result1_ = api.competition_submissions(COMPETITION)[0]
latest_ref1 = str(result1_)  # 最新のサブミット番号
submit_time1 = result1_.date

result2_ = api.competition_submissions(COMPETITION)[1]
latest_ref2 = str(result2_)  # 最新のサブミット番号
submit_time2 = result2_.date


result3_ = api.competition_submissions(COMPETITION)[2]
latest_ref3 = str(result3_)  # 最新のサブミット番号
submit_time3 = result3_.date


result4_ = api.competition_submissions(COMPETITION)[3]
latest_ref4 = str(result4_)  # 最新のサブミット番号
submit_time4 = result4_.date

result5_ = api.competition_submissions(COMPETITION)[4]
latest_ref5 = str(result5_)  # 最新のサブミット番号
submit_time5 = result5_.date

status1 = ''
status2 = ''
status3 = ''
status4 = ''
status5 = ''

while status1 != 'complete' or status2 != 'complete' or status3 != 'complete' or status4 != 'complete' or status5 != 'complete':
    list_of_submission = [api.competition_submissions(COMPETITION)[0],
                          api.competition_submissions(COMPETITION)[1],
                          api.competition_submissions(COMPETITION)[2],
                          api.competition_submissions(COMPETITION)[3],
                          api.competition_submissions(COMPETITION)[4]
                        ]

    for result in list_of_submission:
        if str(result.ref) == latest_ref1:
            status1 = result.status
            result1 = result
        elif str(result.ref) == latest_ref2:
            status2 = result.status
            result2 = result
        elif str(result.ref) == latest_ref3:
            status3 = result.status
            result3 = result
        elif str(result.ref) == latest_ref4:
            status4 = result.status       
            result4 = result
        elif str(result.ref) == latest_ref5:
            status5 = result.status       
            result5 = result 

    status1 = result1.status
    status2 = result2.status
    status3 = result3.status
    status4 = result4.status
    status5 = result5.status

    now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
    elapsed_time1 = int((now - submit_time1).seconds / 60) + 1
    elapsed_time2 = int((now - submit_time2).seconds / 60) + 1
    elapsed_time3 = int((now - submit_time3).seconds / 60) + 1
    elapsed_time4 = int((now - submit_time4).seconds / 60) + 1
    elapsed_time5 = int((now - submit_time5).seconds / 60) + 1

    print("sub1...")
    if status1 == 'complete':
        print('\r', f'run-time: {elapsed_time1} min, LB: {result1.publicScore}')
    else:
        print('\r', f'elapsed time: {elapsed_time1} min', end='')
    print("")

    print("sub2...")
    if status2 == 'complete':
        print('\r', f'run-time: {elapsed_time2} min, LB: {result2.publicScore}')
    else:
        print('\r', f'elapsed time: {elapsed_time2} min', end='')
    print("")

    # print("sub3...")
    # if status3 == 'complete':
    #     print('\r', f'run-time: {elapsed_time3} min, LB: {result3.publicScore}')
    # else:
    #     print('\r', f'elapsed time: {elapsed_time3} min', end='')
    # print("")

    # print("sub4...")
    # if status4 == 'complete':
    #     print('\r', f'run-time: {elapsed_time4} min, LB: {result4.publicScore}')
    # else:
    #     print('\r', f'elapsed time: {elapsed_time4} min', end='')
    # print("")

    # print("sub5...")
    # if status5 == 'complete':
    #     print('\r', f'run-time: {elapsed_time5} min, LB: {result5.publicScore}')
    # else:
    #     print('\r', f'elapsed time: {elapsed_time5} min', end='')
    # print("")
    # print("")
    time.sleep(600)