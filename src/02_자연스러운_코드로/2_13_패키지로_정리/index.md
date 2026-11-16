---
layout: docs
title: "23단계 패키지로 정리"
---

# 23단계 패키지로 정리

지금까지는 단계마다 내용 전체를 파일 하나에 담았습니다. `step01.py`에서 시작하여 `step22.py`까지 도달했죠. 그런데 어느덧 우리의 DeZero는 크게 '성장'했습니다. 그래서 이번 단계에서는 지금까지의 성과를 재사용할 수 있도록 패키지로 정리할 생각입니다.

참고로 파이썬에서는 '모듈', '패키지', '라이브러리'라는 용어를 사용하는데, 보통 다음의 의미로 통용됩니다.

* **모듈**
  모듈은 파이썬 파일입니다. 특히 다른 파이썬 프로그램에서 임포트<sup>import</sup>하여 사용하는 것을 가정하고 만들어진 파이썬 파일을 '모듈'이라고 합니다.

* **패키지**
  패키지는 여러 모듈을 묶은 것입니다. 패키지를 만들려면 먼저 디렉터리를 만들고 그 안에 모듈(파이썬 파일)을 추가합니다.

* **라이브러리**
  라이브러리는 여러 패키지를 묶은 것입니다. 그래서 하나 이상의 디렉터리로 구성되죠. 때로는 패키지를 가리켜 '라이브러리'라고 부르기도 합니다.

---

## 23.1 파일 구성

파일 구성부터 확인하겠습니다. 지금까지는 `step01.py`, `step02.py`, ...처럼 각 step 파일에 코드를 작성했습니다. 이제부터는 이 step 파일 모두에서 DeZero를 이용할 수 있도록 `dezero`라는 공통의 디렉터리를 하나 만들겠습니다. 그래서 최종 파일 구성은 다음과 같습니다.

```text
dezero
  ├── __init__.py
  ├── core_simple.py
  ├── ...
  └── utils.py

steps
  ├── step01.py
  ├── ...
  └── step60.py
```

이와 같이 구성한 뒤 `dezero` 디렉터리에 모듈을 추가하는 것입니다. 그리하여 `dezero`라는 패키지가 만들어지는데, 이 패키지가 바로 우리가 만드는 프레임워크입니다. 앞으로는 주로 이 `dezero` 디렉터리에 있는 파일들에 코드를 추가할 것입니다.

---

## 23.2 코어 클래스로 옮기기

`dezero` 디렉터리에 파일을 추가해보죠. 목표는 이전 단계의 `step22.py` 코드를 `dezero/core_simple.py`라는 코어<sup>core, 핵심</sup> 파일로 옮기는 것입니다. 파일 이름에 `core`를 붙인 이유는 지금까지 구현한 기능들이 DeZero의 핵심이라고 보기 때문입니다. 그리고 뒤에 가서는 최종 형태인 `core.py`로 교체할 계획이라서 당장은 `core_simple.py`로 시작하겠습니다.

그럼 `step22.py`에 정의된 다음 클래스들을 코어 파일로 복사해봅시다.

* `Config`
* `Variable`

* `Function`
* `Add(Function)`
* `Mul(Function)`
* `Neg(Function)`
* `Sub(Function)`
* `Div(Function)`
* `Pow(Function)`

여기에서 `Add(Function)`의 `(Function)`은 `Function` 클래스를 상속했다는 뜻입니다. 보다시피 `Config`, `Variable`, `Function` 클래스가 있고, `Function` 클래스를 상속한 함수(DeZero 함수 클래스)가 여섯 개 있습니다. 이어서 `step22.py`에 정의된 파이썬 함수들도 정리해야 합니다. 즉, 다음 함수들을 코어 파일로 옮길 것입니다.

* `using_config`
* `no_grad`
* `as_array`
* `as_variable`
* `add`
* `mul`
* `neg`
* `sub`
* `rsub`
* `div`
* `rdiv`
* `pow`

처음 두 함수는 DeZero 설정 함수로, 역전파의 활성/비활성을 전환하는 데 사용합니다. 그다음의 `as_array`와 `as_variable`은 인수로 주어진 객체를 `ndarray` 또는 `Variable`로 변환하는 함수입니다. 나머지는 DeZero에서 사용하는 함수입니다. 자, 우선은 `step22.py`에 담긴 클래스와 함수를 그대로 코어 파일에 복사합니다.

> **CAUTION**  
> 지금까지 `Exp` 클래스와 `Square` 클래스 그리고 `exp` 함수와 `square` 함수 등 DeZero에서 사용하는 구체적인 함수들도 구현했습니다. 하지만 이 코드들은 코어 파일에 넣지 않겠습니다. 이 코드들은 나중에 `dezero/functions.py`에 추가할 겁니다.

이제 외부의 파이썬 파일에서 다음과 같이 dezero를 임포트할 수 있습니다.

```python
import numpy as np
from dezero.core_simple import Variable

x = Variable(np.array(1.0))
print(x)
```

**실행 결과**
```
variable(1.0)
```

이와 같이 `from dezero.core_simple import Variable` 줄을 추가하여 `Variable` 클래스를 임포트할 수 있습니다. `dezero.core_simple`처럼 파일 이름까지 명시한 점에 주의하세요. 바로 뒤에서 `core_simple`을 생략하고 `from dezero import Variable`로 사용할 수 있는 구조를 도입할 겁니다.

> **CAUTION**  
> `from ... import ...` 구문을 사용하면 모듈 내의 클래스나 함수 등을 직접 임포트할 수 있습니다. 또한 `import XXX as A`라고 쓰면 `XXX`라는 모듈을 `A`라는 이름으로 임포트할 수 있습니다. 예를 들어 `import dezero.core_simple as dz`로 쓰면 `dezero.core_simple` 모듈을 `dz`라는 이름으로 임포트합니다. 그런 다음 `Variable` 클래스를 사용하려면 `dz.Variable`이라고 쓰면 됩니다.

---

## 23.3 연산자 오버로드

이것으로 `step22.py`의 코드 대부분이 옮겨졌습니다. 이제부터는 오버로드한 연산자들을 `dezero`로 옮기겠습니다. 이를 위해 코어 파일인 `dezero/core_simple.py`에 다음 함수들을 추가합니다.

**dezero/core_simple.py**
```python
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
```

```python
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
```

`setup_variable`은 `Variable`의 연산자들을 오버로드해주는 함수입니다. 이 함수를 호출하면 `Variable`의 연산자들이 설정됩니다. 그렇다면 이 함수는 어디에서 호출하면 좋을까요? 바로 `dezero/__init__.py` 파일입니다.

`__init__.py`는 모듈을 임포트할 때 가장 먼저 실행되는 파일입니다. 우리의 경우 `dezero` 패키지에 속한 모듈을 임포트할 때 `dezero/__init__.py`의 코드가 첫 번째로 호출됩니다. 그래서 `dezero/__init__.py`에 다음 코드를 작성해 넣어야 합니다.

**dezero/\_\_init\_\_.py**
```python
from dezero.core_simple import Variable
from dezero.core_simple import Function
from dezero.core_simple import using_config
from dezero.core_simple import no_grad
from dezero.core_simple import as_array
from dezero.core_simple import as_variable
from dezero.core_simple import setup_variable

setup_variable()
```

이와 같이 `setup_variable` 함수를 임포트해 호출하도록 합니다. 이렇게 함으로써 `dezero` 패키지를 이용하는 사용자는 반드시 연산자 오버로드가 이루어진 상태에서 `Variable`을 사용할 수 있습니다.

한편 `__init__.py`의 시작이 `from dezero.core_simple import Variable`인데, 이 문장이 실행됨으로써 `dezero` 패키지에서 `Variable` 클래스를 곧바로 임포트할 수 있습니다. 예를 들어 다음과 같이 이용할 수 있습니다.

```python
# dezero를 이용하는 사용자의 코드

# from dezero.core_simple import Variable
from dezero import Variable
```

즉, 지금까지 `from dezero.core_simple import Variable`이라고 작성한 것을 `from dezero import Variable`처럼 짧게 줄일 수 있습니다. 마찬가지로 `dezero/__init__.py`의 임포트문들 덕분에 사용자는 나머지 `Function`이나 `using_config` 등도 '간소화된' 임포트를 이용할 수 있게 됩니다.

---

## 23.4 실제 \_\_init\_\_.py 파일

이 교제에서는 앞으로(23단계에서 32단계까지) DeZero 코어 파일로 `dezero/core_simple.py`를 사용합니다. 그러다가 33단계부터는 `dezero/core.py`로 대체할 것입니다. 그래서 실제 `dezero/__init__.py`는 `core_simple.py`와 `core.py` 중 하나를 선택해 임포트하도록 작성되어 있습니다.

**dezero/\_\_init\_\_.py**
```python
is_simple_core = True

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
else:
    from dezero.core import Variable
    from dezero.core import Function
    ...

setup_variable()
```

이와 같이 `is_simple_core` 플래그로 임포트할 대상을 선택합니다. `is_simple_core`가 `True`면 `core_simple.py`에서, `False`면 `core.py`에서 임포트가 이루어집니다.

> **CAUTION**  
> 코드를 실습해보려면 `is_simple_core` 플래그값을 적절히 수정하여 사용하기 바랍니다. 32단계까지는 `True`로 설정하고, 33단계부터는 `False`로 바꿔주면 됩니다.

---

## 23.5 dezero 임포트하기

이렇게 하여 `dezero`라는 패키지가 만들어졌습니다. 이제 이번 단계용 `step23.py`는 다음과처럼 작성할 수 있습니다.

**steps/step23.py**
```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
```

**실행 결과**
```
variable(16.0)
8.0
```

우선 `if '__file__' in globals():` 문장에서 `__file__` 이라는 전역 변수가 정의되어 있는지 확인합니다. `python steps/step23.py`처럼 터미널에서 `python` 명령으로 실행한다면 `__file__` 변수가 정의되어 있습니다. 이 경우 현재 파일(`step23.py`)이 위치한 디렉터리의 부모 디렉터리(`..`)를 모듈 검색 경로에 추가합니다. 이로써 파이썬 명령어를 어디에서 실행하든 `dezero` 디렉터리의 파일들은 제대로 임포트할 수 있게 됩니다. 예를 들어 명령줄에서 `python steps/step23.py` 형태로 실행하든 `cd steps; python step23.py` 형태로 디렉터리를 옮겨 실행하

든 상관없이 코드가 정상 작동합니다. 참고로 교제 지면에서는 편의상 이 모듈 검색 경로 추가 코드는 (매번 똑같이 반복되므로) 생략하고 보여줍니다.

> **NOTE**  
> 검색 경로 추가 코드는 현재 개발 중인 `dezero` 디렉터리를 임포트하기 위해 일시적으로 사용하는 것입니다. (예를 들어 `pip install dezero` 등의 명령으로) DeZero가 패키지로 설치된 경우라면 DeZero 패키지가 파이썬 검색 경로에 추가됩니다. 따라서 앞에서와 같이 경로를 수동으로 추가하는 일은 필요치 않게 됩니다. 또한 `__file__` 변수는 파이썬 인터프리터의 인터랙티브 모드와 구글 콜랩<sup>Google Colab</sup>\* 등의 환경에서 실행하는 경우에는 정의되어 있지 않습니다. 이 점을 고려하여 (step 파일을 수정 없이 구글 콜랩에서도 동작하도록 하기 위해) 부모 디렉터리를 검색 경로에 추가할 때 `if '__file__' in globals():` 라는 조건 검사 문장을 넣었습니다.

방금 보여드린 코드가 `step23.py`의 전부입니다(생략한 코드는 없습니다). 이것으로 DeZero 프레임워크의 원형이 완성되었습니다. 앞으로는 `dezero` 디렉터리에 있는 파일(모듈)들을 확장하는 식으로 진행하겠습니다.

---
\* 옮긴이_ 번역 시점에 우리나라에서는 '구글 코랩'으로 더 많이 쓰고 있으나, 구글의 공식 소개 동영상(https://bit.ly/31hbc1L)에서도 '콜랩(Colab)'으로 표기하고 있고, 실제 원어민 발음에 훨씬 가까우며, 외래어 표기법상으로도 맞기 때문에 이 교제에서는 '콜랩'으로 옮기겠습니다.
