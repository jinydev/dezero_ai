# 🧠 DeZero: 밑바닥부터 시작하는 딥러닝 프레임워크

<p align="center">
  <a href="https://dezero.ai.jiny.dev">
    <img src="https://img.shields.io/badge/Site-dezero.ai.jiny.dev-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Live Site" />
  </a>
  <a href="https://github.com/jinydev/dezero_ai">
    <img src="https://img.shields.io/badge/GitHub-jinydev%2Fdezero__ai-black?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/Jekyll-4.4.1-CC0000?style=for-the-badge&logo=jekyll&logoColor=white" alt="Jekyll 4.4" />
</p>

---

## 💡 프로젝트 소개

**DeZero(밑바닥부터 시작하는 딥러닝 3)**는 파이썬 기본 문법과 넘파이(NumPy)만을 사용하여, 현대 딥러닝 프레임워크(PyTorch, TensorFlow)의 핵심 아키텍처를 밑바닥부터 직접 설계하고 구현해보는 강의 교재 웹사이트입니다.

프레임워크를 단순 '사용자' 입장에서 다루는 것을 넘어, **"엔진 내부가 실제로 어떻게 회전하는가?"**를 직접 코딩하며 배웁니다. 단 3줄의 `Variable` 클래스에서 출발하여, 자동 미분(Define-by-Run) 엔진, 연산자 오버로딩, 메모리 최적화, 고차 미분, 신경망 추상화 계층, 그리고 GPU 가속과 CNN/RNN/LSTM에 이르기까지 60단계에 걸친 대장정을 완성합니다.

- 🌐 **공식 온라인 강의 사이트**: [https://dezero.ai.jiny.dev](https://dezero.ai.jiny.dev)
- 📦 **소스코드 저장소**: [https://github.com/jinydev/dezero_ai](https://github.com/jinydev/dezero_ai)

---

## 🗺️ 5대 고지 탐험 로드맵 (Curriculum)

| 고지 | 명칭 | 핵심 학습 내용 및 구현 목표 | 단계 |
| :---: | :--- | :--- | :---: |
| **00** | **프롤로그** | 프레임워크를 직접 만들어야 하는 이유, DeZero 3대 특징(Define-by-Run, 모듈성, 간결함), 학습 환경 설정 | 준비 |
| **01** | **제1고지: 미분 자동 계산** | `Variable`, `Function` 기본 클래스 설계, 계산 그래프(DAG) 형성, 수치 미분 검증, 연쇄 법칙 기반 자동 역전파 엔진 완성 | 1~10단계 |
| **02** | **제2고지: 자연스러운 코드로** | 가변 길이 인수 지원, 파이썬 사칙연산자 오버로딩(`+`, `-`, `*`, `/`), `weakref` 순환 참조 메모리 관리, `no_grad` 모드 | 11~24단계 |
| **03** | **제3고지: 고차 미분 계산** | 역전파 계산 자체를 계산 그래프로 연결, 2차/고차 미분(Double Backprop) 구현, 테일러 급수 근사, 뉴턴 방법 최적화 | 25~36단계 |
| **04** | **제4고지: 신경망 만들기** | 다차원 텐서 연산, `Parameter`, `Layer`, `Model` 객체지향 추상화, 활성화 함수, 손실 함수, 옵티마이저(SGD, Adam), 분류/회귀 | 37~51단계 |
| **05** | **제5고지: DeZero의 도전** | CuPy 연동 NVIDIA GPU 가속, 가중치 저장 및 복원, 드롭아웃, 2D 합성곱(`Conv2d`) & 풀링(`Pooling`), VGG16, 시계열(RNN, LSTM) | 52~60단계 |
| **06** | **부록 (Appendix)** | 인플레이스 연산 메모리 기법, `get_item` 슬라이싱 구현, 구글 코랩(Colab) 실행 가이드, API 레퍼런스 | 보충 |

---

## 👥 도로시 유니버스 캐릭터 가이드

본 교재에는 학습자의 이해를 돕고 난해한 수학적 개념을 직관적으로 풀어나가기 위해 **도로시 탐험대** 캐릭터들이 교육용 2D 삽화(스튜디오 지브리 스타일)로 함께합니다.

- 👨‍🏫 **이호진 교수님 (Prof. Hojin)**: 딥러닝과 소프트웨어 공학의 원리를 통찰력 있게 이끌어주는 든든한 멘토
- 👧 **도로시 (Dorothy)**: 호기심과 열정으로 DeZero 엔진을 직접 조립해 나가는 주니어 메이커
- 🐱 **조교 지니 (Jiny)**: 턱시도 코트를 입고 꼼꼼한 코드 리뷰와 타입/메모리 검증을 맡은 수석 연구실 냥이
- 🐶 **토토 (Toto)**: 언제나 곁에서 긍정 에너지를 불어넣어 주는 순백의 비타민 반려견
- 🤖 **틴맨 (Tinny)**: 기어와 센서로 무장하고 계산 그래프와 텐서 연산을 번개처럼 돕는 AI 메이커 로봇

---

## 🛠️ 프로젝트 디렉터리 구조

```text
A04_Dezero/
├── _config.yml               # Jekyll 메인 설정 (URL, kramdown, mathjax 등)
├── Gemfile                   # Ruby 의존성 패키지 정의
├── README.md                 # 프로젝트 종합 안내 문서 (본 문서)
├── guides/                   # 강의 교재 스타일 및 생성형 AI 이미지 제작 가이드
│   ├── 생성형_이미지.md
│   └── 커버생성_가이드.md
├── src/                      # Jekyll 소스 디렉터리
│   ├── _data/                # 네비게이션 YAML 데이터 (part0.yml ~ part6.yml 등)
│   ├── _includes/            # 공통 헤더, 네비게이션, 브레드크럼, 푸터 템플릿
│   ├── _layouts/             # 문서 및 블로그 기본 레이아웃
│   ├── assets/               # CSS, JS, 폰트 및 공통 이미지
│   ├── 00_프롤로그/
│   ├── 01_미분_자동_계산/
│   ├── 02_자연스러운_코드로/
│   ├── 03_고차_미분_계산/
│   ├── 04_신경망_만들기/
│   ├── 05_DeZero의_도전/
│   └── 06_부록/
└── docs/                     # GitHub Pages 배포용 정적 빌드 결과물 (배포 디렉터리)
```

---

## 🚀 로컬 환경 실행 및 사이트 빌드 가이드

### 1. 사전 요구사항 (Prerequisites)
- **Ruby**: 3.0 이상 권장
- **Bundler**: `gem install bundler`
- **Python**: 3.8 이상 (DeZero 실습 및 Pillow 이미지 검증용)

### 2. 의존성 설치
```bash
bundle install
```

### 3. 사이트 빌드 (정적 배포 파일 생성)
```bash
bundle exec jekyll build
```
빌드된 결과물은 `docs/` 디렉터리에 생성되어 GitHub Pages와 직접 연동됩니다.

### 4. 로컬 개발 서버 실행 (라이브 리로드)
```bash
bundle exec jekyll serve --livereload --force_polling --port 4000
```
웹 브라우저에서 `http://localhost:4000`으로 접속하여 실시간 변경 사항을 확인할 수 있습니다.

---

## 🌐 배포 및 커스텀 도메인

본 사이트는 GitHub Pages의 배포 경로를 `/docs` 폴더로 지정하여 호스팅되며, CNAME 레코드를 통해 전용 도메인으로 서비스됩니다:

- **도메인**: `https://dezero.ai.jiny.dev`
- **설정 파일**: [src/CNAME](file:///Users/hojin9/Library/Mobile%20Documents/com~apple~CloudDocs/02.강의/A04_Dezero/src/CNAME) (`dezero.ai.jiny.dev`)

---

## ✍️ 저자 및 문의

- **저자 / 강사**: 이호진 (Hojin Lee)
- **Email**: infohojin@gmail.com
- **Website**: [https://jiny.dev](https://jiny.dev)
- **Copyright**: © 2026 Hojin Lee. All Rights Reserved.
