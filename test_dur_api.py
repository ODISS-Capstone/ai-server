import requests
from urllib.parse import unquote

# 1. API 접속 정보 설정
url = 'http://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03'

# [중요] 403 에러 방지: 인증키는 반드시 '디코딩된 키(Decoded)'를 사용하세요.
# 만약 아래 키가 작동하지 않는다면, 포털에서 새로 발급받은 'Decoding' 키를 붙여넣으세요.
raw_service_key = 'aeccf16075bf1f0e9a79588ee6c0da4c5dd0a831fc0f3861742c6cff11f6cd59'
service_key = unquote(raw_service_key)

# 2. 요청 파라미터 (제공해주신 항목 반영)
params = {
    'serviceKey': service_key,
    'pageNo': '1',                # 페이지 번호
    'numOfRows': '10',            # 한 페이지 결과 수
    'type': 'json',               # 응답 형식 (xml/json)
    'item_name': '타이레놀',        # 품목명 (검색어)
    # 'entp_name': '',            # 업체명 (필요시 입력)
    # 'item_seq': '',             # 품목일련번호 (필요시 입력)
    # 'edi_code': '',             # 보험코드 (필요시 입력)
}

try:
    # 3. API 호출
    response = requests.get(url, params=params, timeout=10)
    
    # 4. 결과 출력
    if response.status_code == 200:
        data = response.json()
        
        # 실제 데이터 접근 (OpenAPI 표준 구조)
        if 'body' in data and 'items' in data['body']:
            items = data['body']['items']
            print(f"--- 검색 결과: {data['body']['totalCount']}건 중 {len(items)}건 표시 ---")
            
            for item in items:
                print(f"[품목명] : {item.get('ITEM_NAME')}")
                print(f"[업체명] : {item.get('ENTP_NAME')}")
                print(f"[보험코드]: {item.get('EDI_CODE', '정보없음')}")
                print(f"[색상]   : {item.get('COLOR_CLASS1')}")
                print(f"[모양]   : {item.get('DRUG_SHAPE')}")
                print(f"[이미지] : {item.get('ITEM_IMAGE')}")
                print("-" * 50)
        else:
            print("결과가 없습니다. 파라미터를 확인해주세요.")
            print("응답 내용:", data)
            
    elif response.status_code == 403:
        print("Error 403: 접근 거부! (인증키 문제일 확률 99%)")
        print("1. 인증키가 '디코딩(Decoding)'된 것인지 확인하세요.")
        print("2. API 활용 신청 후 '승인' 상태인지 확인하세요 (동기화에 1~2시간 소요).")
    else:
        print(f"에러 발생: {response.status_code}")

except Exception as e:
    print(f"연결 실패: {e}")