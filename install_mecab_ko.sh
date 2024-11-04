#!/bin/bash

# 스크립트 실행 중 오류 발생시 즉시 중단
set -e

echo "시스템 기본 도구 설치 여부를 확인합니다..."

# 필수 도구 목록
REQUIRED_PACKAGES=(
    "wget"
    "curl"
    "git"
    "build-essential"
    "autoconf"
    "automake"
    "libtool"
    "pkg-config"
    "python3"
    "python3-pip"
)

# 설치된 패키지 확인 및 미설치된 패키지 설치
PACKAGES_TO_INSTALL=()
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! command -v $package >/dev/null 2>&1; then
        if ! dpkg -l | grep -q "^ii  $package "; then
            PACKAGES_TO_INSTALL+=("$package")
        fi
    fi
done

# 필요한 패키지가 있다면 설치
if [ ${#PACKAGES_TO_INSTALL[@]} -ne 0 ]; then
    echo "다음 패키지들을 설치해야 합니다: ${PACKAGES_TO_INSTALL[@]}"
    echo "패키지 관리자 업데이트 중..."
    apt update
    
    echo "필요한 패키지 설치 중..."
    apt install -y "${PACKAGES_TO_INSTALL[@]}"
fi

# 작업 디렉토리 생성 및 이동
WORK_DIR="mecab_install_$(date +%Y%m%d_%H%M%S)"
mkdir -p $WORK_DIR
cd $WORK_DIR

echo "MeCab 다운로드 및 설치 중..."
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
make install
ldconfig  # 시스템에 라이브러리 경로 설정
cd ..

echo "MeCab 한국어 사전 다운로드 및 설치 중..."
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720

# 사전 설치 구성 및 빌드
autoreconf --install
./configure
make
make install
cd ..

echo "KoNLPy용 MeCab 설치 스크립트 실행 중..."
curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash

echo "Python용 MeCab 패키지 설치 중..."
pip3 install mecab-python3

echo "설치 테스트를 실행합니다..."
python3 - << EOF
import MeCab

# MeCab 형태소 분석기 초기화
mecab = MeCab.Tagger()

# 테스트 문장
test_sentence = "안녕하세요. MeCab 설치가 잘 되었습니다."
print("테스트 문장: ", test_sentence)

# 형태소 분석 결과 출력
result = mecab.parse(test_sentence)
print("형태소 분석 결과:\n", result)
EOF

# 작업 디렉토리 정리
cd ..
rm -rf $WORK_DIR

echo "MeCab 설치 및 테스트가 완료되었습니다."

# 사용 방법:
# 1. 이 스크립트를 install_mecab_ko.sh로 저장
# 2. 실행 권한 부여: chmod +x install_mecab_ko.sh
# 3. 실행: bash ./install_mecab_ko.sh
# 4. 관리자 권한 필요한 경우에는 sudo ./install_mecab_ko.sh 로 실행