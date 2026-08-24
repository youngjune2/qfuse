"""
retrieve_hybrid.py — HybridRAG 베이스라인의 'KG 반쪽'(순수 정형 Text2Cypher).

베이스라인 정체성: 정형(origin='structured') 엣지만 본다. document_link/extracted 는
스키마에서 아예 제외 -> 문서에서 추출된 브리지 홉(엔티티↔엔티티)을 구조적으로 traverse
할 수 없다. 이 한계가 HybridRAG 가 벡터 반쪽에 의존하게 만드는 지점이고, 두 결과를 그냥
이어 붙이는(naive concat) 방식의 한계를 드러내는 대조군이다.

'바닐라'의 의미(사용자 선택): 프루닝 없음(전체 정형 스키마 통짜) · gloss 없음 · dedup 없음
· 브리지 recall 바닥 없음. 단, Cypher 생성/정적검증/자동수정/재시도 같은 '실행 견고성'
기계는 Strategy D 와 공유한다(두 방식이 같은 실행기를 쓰게 해 비교를 공정하게 유지 —
차이를 '스키마/검색 전략'으로 격리하려는 것이지 '실행기 품질'로 만들려는 게 아니다).
"""

from __future__ import annotations

from retrieve_d import SubgraphText2CypherRetriever


class VanillaStructuredRetriever(SubgraphText2CypherRetriever):
    """정형 origin 엣지만, 전체 스키마 통짜 Text2Cypher. 프루닝/gloss/dedup/브리지 바닥 off."""

    strategy_name = "KG-vanilla-structured"

    def __init__(self, graph, llm, catalog, tau: float = 0.5, retries: int = 2,
                 debug: bool = False):
        # 부모 초기화: 프루닝 off + dedup off + (extracted 지지도 필터를 사실상 무력화).
        super().__init__(graph, llm, catalog, tau=tau, retries=retries, debug=debug,
                         min_extracted_support=10 ** 9, prune_mode="none",
                         dedup_structured=False)
        # 카탈로그를 정형 origin 만으로 좁힌다. 이 뒤로 all_types/valid_* 전부 정형만 남는다.
        struct = [x for x in self.rel_catalog if x[3] == "structured"]
        self.rel_catalog = struct
        self.all_types = sorted({x[1] for x in struct})       # prune_mode="none" 이 그대로 스키마로
        self.backbone_types = list(self.all_types)
        self.extracted_types = []
        self.type_support = {}
        self.rel_gloss = {}                                    # gloss 없음
        # 정적 검증 인덱스도 정형만 -> 모델이 extracted/문서 관계를 쓰면 '존재하지 않음'으로 거부됨.
        self.valid_patterns = {(a, t, b) for (a, t, b, o, c) in struct}
        self.valid_labels = {x[0] for x in struct} | {x[2] for x in struct}
        if debug:
            print(f"  [KG-vanilla] 정형 관계타입 {len(self.all_types)}종 / "
                  f"패턴 {len(self.valid_patterns)} / 라벨 {len(self.valid_labels)}")
