# semantic_transmission 모델 구조

현재 `semantic_transmission`의 ETRI 실행 설정인 **SKEM + DSA 구조**입니다.
송신단에서 의미 정보를 추출하고, 두 전송 경로를 거쳐 수신단에서 영상을 생성합니다.

```mermaid
flowchart TB
    subgraph TX["① 송신단 — Semantic Encoder"]
        X["원본 영상"] --> P["영상 전처리<br/>576 × 320 · 24 fps"]
        P --> S["SKEM · InternVL2-8B + PSSS<br/>의미 변화에 따른 키프레임 선택 · 구간 분할"]

        S -->|"키프레임"| E["NTSCC Encoder<br/>시각 정보 → 복소 심벌"]
        S -->|"영상 구간"| C["PLLaVA-7B<br/>구간별 텍스트 설명"]
        S -->|"영상 구간"| F["UniMatch<br/>광류 → 움직임 크기 요약"]

        C --> M["메타데이터 패킷 구성"]
        F --> M
        S -->|"키프레임 위치 · 영상 정보"| M
        E -.->|"rate index · 정규화 정보"| M
        M --> L["LDPC Encoder<br/>16-QAM 변조"]
    end

    subgraph CH["② 통신 채널 — AWGN 시뮬레이션 · SNR 10 dB"]
        A["시각 정보 경로<br/>연속 복소 JSCC 심벌"]
        B["디지털 정보 경로<br/>캡션 · 움직임 · 메타데이터"]
    end

    E --> A
    L --> B

    subgraph RX["③ 수신단 — Semantic Decoder"]
        D["NTSCC Decoder<br/>키프레임 복원"]
        R["16-QAM 복조 · LDPC Decoder<br/>메타데이터 복원"]
        R -.->|"rate index · 정규화 정보"| D

        subgraph GEN["DSA + Open-Sora v1.2"]
            V["VAE Encoder<br/>키프레임 잠재표현"]
            T["T5 Text Encoder<br/>캡션 + motion score"]
            Q["DSA<br/>구간 길이 · 잠재 크기 조절<br/>키프레임 조건 · 시간축 정렬"]
            G["STDiT3<br/>조건부 확산 생성"]
            W["VAE Decoder<br/>영상 구간 복원"]

            V --> Q
            Q --> G
            T --> G
            G --> W
            W -.->|"다음 구간의 참조"| Q
        end

        D --> V
        R -->|"텍스트 · 움직임 크기"| T
        R -->|"키프레임 위치 · 영상 정보"| Q
        W --> O["겹침 제거 · 구간 연결<br/>최종 복원 영상"]
    end

    A --> D
    B --> R

    classDef source fill:#edf4ff,stroke:#426da9,color:#162c49;
    classDef channel fill:#fff3df,stroke:#bb852b,color:#513a12;
    classDef receiver fill:#eaf5ee,stroke:#4b8b66,color:#183e28;
    class X,P,S,E,C,F,M,L source;
    class A,B channel;
    class D,R,V,T,Q,G,W,O receiver;

    style TX fill:#f8faff,stroke:#426da9
    style CH fill:#fffbf3,stroke:#bb852b
    style RX fill:#f6fbf7,stroke:#4b8b66
    style GEN fill:#ffffff,stroke:#4b8b66,stroke-dasharray:5 5
```

*그림 1. 현재 구현된 LGVSC 기반 영상 의미 통신 시스템. 점선은 복호 보조정보 전달과 구간 간 참조를 나타냅니다.*

- **송신 데이터:** 키프레임의 NTSCC 심벌과 캡션·움직임 크기·복원에 필요한 메타데이터입니다.
- **움직임 표현:** UniMatch 광류를 **구간당 스칼라 하나**로 요약하고, 수신단에서 `motion score`라는 텍스트 조건으로 사용합니다.
- **DSA의 역할:** 구간마다 생성 길이와 키프레임 조건을 정렬하고, 이전 생성 구간을 참조하도록 구성합니다.

구현 근거: [실행 설정](../configs/etri_official.json),
[송수신 코드](../src/semantic_transmission/codec_transport.py),
[생성 디코더](../04_semantic_decoder/scripts/mydemo_new_align_sh.py).
