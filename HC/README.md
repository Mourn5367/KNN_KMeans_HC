# 계층적 클러스터링 (Hierarchical Clustering)

이 프로젝트는 Single-Linkage 방식을 사용한 계층적 클러스터링 알고리즘을 처음부터 직접 구현한 파이썬 스크립트를 포함합니다.

## 파일 설명

- `hierarchical_clustering_colab.py`: 메인 파이썬 스크립트입니다. Google Colab 환경에 최적화되어 있습니다.
- `dendrogram.png`: 위 스크립트 실행 시 생성되는 덴드로그램 결과 이미지 예시입니다.

## 주요 기능

- **덴드로그램 시각화**: Matplotlib을 사용하여 병합 과정, 거리, 클러스터별 색상 등을 직접 그려주는 `enhanced_dendrogram` 함수를 포함합니다.
  - 주요 클러스터(Major Cluster) 개수를 지정하여 해당 클러스터의 가지들을 다른 색상으로 강조할 수 있습니다.

## 사용법 (Google Colab)

1. `hierarchical_clustering_colab.py` 파일의 모든 코드를 코랩의 한 셀에 복사하여 붙여넣고 실행합니다.
2. 다음 셀에서 아래 코드를 입력하고 실행하여 예제를 시작합니다.
   ```python
   run_hc_example()
   ```
3. 덴드로그램 그래프가 화면에 출력되고, 이미지 파일로도 저장됩니다.


## 플로우차트

![플로우차트](https://github.com/Mourn5367/KNN_KMeans_HC/blob/main/HC/HC_flow.jpg?raw=true)
