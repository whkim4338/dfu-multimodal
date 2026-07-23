"""
dfu_multimodal — 이미지(frozen DINOv3) + 임상 데이터 멀티모달 실험, fusion 방식 교체 가능.

`dfu_fusion`(실험 4: DINOv3 + MLP concat fusion)의 구조와 코드 스타일을 계승하되, fusion
방식 자체를 실험 변수로 다룬다. `--fusion-strategy` 인자로 concat / gated / gbdt 등을
선택해 동일한 데이터·태스크·평가 지표 위에서 fusion 방식의 성능을 비교한다.

amputation_risk는 `dfu_fusion`과 동일한 이유(라벨링 근거 애매, 예측 실익 불확실)로 예측
대상에서 제외했다. Wagner grade(CORAL ordinal) / SINBAD 6개 component(binary)만 예측한다.

세부 계획은 Project.md 참고. 다른 dfu_* 패키지를 import하지 않는 완전 독립 패키지다.
"""

__version__ = "0.1.0"
