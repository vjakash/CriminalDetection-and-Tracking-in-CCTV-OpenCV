import requests
import json

URL1 = 'https://www.sms4india.com/api/v1/createSenderId'
# post request
def sendPost(reqUrl, apiKey, secretKey, useType, senderId):
  req_params = {
  'apikey':apiKey,
  'secret':secretKey,
  'usetype':useType,
  'senderid':senderId
  }
  return requests.post(reqUrl, req_params)
#responsesender = sendPost(URL1,'GT1SN34NZSHPT3RNG6GNR59AVJ8VX9Z6','O01H9UJYTX91FL3U','prod','887070')
responsesender = sendPost(URL1,'1IJN4HK99X3M44G68XPANYUDVSY40C26','KHPF9D7BM7U37C94','prod','887070')
print (responsesender.text)
