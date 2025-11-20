## 추천 Git Flow 전략: Simplified Git Flow (Feature Branching)

이 전략은 각 분석 테마를 독립적인 기능(Feature)으로 보고, 안전하고 체계적으로 진행하는 데 초점을 맞춥니다.

### 1. 핵심 브랜치 구조

| 브랜치 이름 | 목적 | 관리 주체 |
| :--- | :--- | :--- |
| **`main`** (또는 `master`) | **배포 가능한 안정적인 버전** (Production-ready). 최종 보고서 및 시각화 화면의 **최종 제출본**만 포함합니다. 이 브랜치에서는 직접 작업하지 않습니다. | 교수님에게 제출되는 **최종 결과물**. |
| **`develop`** | **통합 브랜치**. 모든 개발 작업이 합쳐지는 곳입니다. 주간/격주 단위의 모든 분석 결과를 **안정적으로 통합**합니다. | 현재까지의 **최신 진행 상황**. |
| **`feature/주제명`** | **기능(테마) 개발 브랜치**. 특정 분석 테마(예: `feature/seoul_total_pattern`, `feature/gu_comparison`)를 진행하는 독립적인 브랜치입니다. | **주간/격주 단위의 개별 분석 작업**. |

### 2. 작업 흐름 (Workflow)

1.  **시작**: `main` 브랜치에서 **`develop`** 브랜치를 생성합니다. (`git branch develop`, `git checkout develop`)
2.  **테마 시작 (주간/격주)**:
    * `develop` 브랜치에서 새로운 기능 브랜치(테마)를 생성하고 체크아웃합니다.
    * 예: `git checkout -b feature/gu_comparison develop`
3.  **분석 및 개발**:
    * `feature/gu_comparison` 브랜치에서 Jupyter Notebook 파일(`.ipynb`)을 작성하고 분석 코드를 실행하며, **Markdown 셀로 메모 및 결과를 정리**합니다.
    * 수시로 커밋하여 작업 기록을 남깁니다. (`git commit -am "구별 평균 비교 분석 완료"`)
4.  **통합 (Merge)**:
    * 분석 테마가 완료되면, `feature/gu_comparison` 브랜치를 **`develop`** 브랜치로 병합합니다. **Pull Request (PR) 또는 Merge Request**를 통해 병합을 진행하는 것을 강력히 추천합니다. (혼자 하는 프로젝트라도 PR을 작성하면 작업 내용 정리가 잘 됩니다.)
    * `git checkout develop`
    * `git merge feature/gu_comparison`
    * `feature/gu_comparison` 브랜치는 삭제합니다.
5.  **반복**: 다음 테마를 위해 다시 **Step 2**로 돌아가서 새로운 `feature/테마명` 브랜치를 생성하고 작업을 반복합니다.
6.  **최종 제출**: 학기말에 `develop` 브랜치가 최종 완성되었다고 판단되면, **`develop`**을 **`main`** 브랜치로 병합하고 태그를 붙여 최종 제출본임을 명시합니다. (`git checkout main`, `git merge develop`, `git tag v1.0.0`)

- **브랜치 관리** : VS Code 하단 파란색 상태 표시줄을 클릭하면 **브랜치 생성/전환**이 매우 쉽게 가능합니다. `feature/` 브랜치 간의 전환이 간편해집니다.
- **Commit & Push** : VS Code 좌측 소스 제어(Source Control) 탭에서 변경 사항을 확인하고 **스테이징, 커밋, 푸시**를 GUI로 진행할 수 있습니다. 코드를 실행할 때마다 혹은 중요한 결과가 나올 때마다 커밋을 습관화합니다.

- **tools**
- git lens
- (git graph)
- (source Control)

- tortoisGit

## git commit convention

(chat gpt)[https://chatgpt.com/g/g-68137a65261481918359aee65afeacd9-git-commit-assistant/c/691b1bf9-9978-8321-bbb6-1c67ce2ada91]

## 명령어

### merge
- a 브랜치에서 커밋
git add .
git commit -m "message"
(git push origin a)

- develop 브랜치로 이동
git checkout develop
(git pull origin develop)

- deveolp에 a 브랜치를 병합
git merge a 

(충돌 있다면...)
(git add .)
(git commit)

(git push origin develop)

(브랜치를 삭제?!?!)
(git branch -d a)
(git push origin --delete a)

### history
git log --graph

### 터미널 출력을 txt로
PYTHONIOENCODING=utf-8 python all_of_gu_price_cycle_detection.py > results/analysis_results.txt
덮어쓰기: >, 내용추가: >>