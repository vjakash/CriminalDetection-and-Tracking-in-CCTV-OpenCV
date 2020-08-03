import requests
def sendsms(msg):
    url = "https://www.fast2sms.com/dev/bulk"
    payload = "sender_id=FSTSMS&message="+msg+"&language=english&route=p&numbers=8870631234,7338987648"
    headers = {
    'authorization': "j39GYkoNaKLegEOTtxSIvyfupC4brAUWXHFJlmqh7Q1RB8sci2dLy2Ge5pNWb6CIORh7rV1zYcwUxfuo",
    'Content-Type': "application/x-www-form-urlencoded",
    'Cache-Control': "no-cache",
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    print(response.text)
    
sendsms("hi from test")
