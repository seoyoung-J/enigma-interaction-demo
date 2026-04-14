# ENIGMA-51 Industrial Interaction Demo

---

ENIGMA-51 데이터셋 기반 산업 작업 상호작용 분류 및 예측 데모입니다. 최근 4개 프레임의 feature를 입력으로 사용해 다음을 예측합니다.

- 상호작용 유무
- 현재 pair
- 미래 pair

## Dataset
- ENIGMA-51
- Paper: Ragusa et al., *ENIGMA-51: Towards a Fine-Grained Understanding of Human Behavior in Industrial Scenarios*, WACV 2024

## Features
- 최근 4개 프레임 기준 시퀀스 입력
- 상호작용 유무 분류
- 현재 pair 분류
- 미래 pair 예측
- LSTM / GRU / BiLSTM 기반 모델 비교
- Streamlit 기반 결과 확인

## Project Structure
- `app.py` : Streamlit 데모 앱
- `notebook/` : 실험 및 모델링 노트북
- `requirements.txt` : 실행 환경 패키지 목록

## Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py  
```

## Note
- 대용량 데이터, feature, 모델 파일은 본 저장소에 포함하지 않습니다. 실행을 위해서는 별도의 데이터 및 학습 산출물이 필요합니다.