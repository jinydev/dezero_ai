---
layout: docs
title: "2단계 변수를 낳는 함수"
---

# 2단계 변수를 낳는 함수

![1_2_변수를_낳는_함수](./1_2_변수를_낳는_함수.png)
**그림 02-Intro** 입력 변수를 받아 새로운 출력 변수를 창조하는 변환 장치 Function과 Square 구현 (이호진 교수와 도로시 탐험대)

---


앞 단계에서 Variable 클래스를 상자로 사용할 수 있게 했습니다. 

하지만 지금 이대로는 그냥 상자일뿐입니다. 우리에겐 단순한 상자를 마법의 상자로 바꾸는 장치가 필요한데, 그 열쇠는 바로 '함수'입니다. 

![이호진 교수님이 들려주는 마법의 열쇠 함수(Function)의 개념](./img/fig_02_01_intro_function_key.jpg)  
*이호진 교수님이 단순한 상자를 마법의 상자로 변환시키는 '함수(Function)'라는 황금 열쇠를 소개하고 있습니다.*

---

이번 단계에서는 함수에 대해 생각해보겠습니다.

## 2.1 함수란

함수란 무엇일까요? 

조금 딱딱하게 표현하면 '어떤 변수로부터 다른 변수로의 대응 관계를 정한 것'이라고 할 수 있습니다. 

![함수 f를 통한 입력 변수 x와 출력 변수 y의 대응 관계 파이프라인](./img/fig_02_02_function_mapping_pipeline.jpg)  
*입력 변수 $x$ 상자가 함수 $f$ 파이프라인을 통과하여 새로운 출력 변수 $y$ 상자로 변환되는 대응 관계를 보여줍니다.*

---

수학적에서의 함수와 컴퓨터에서의 함수가 어떻게 다른지 한번 생각해 보도록 하겠습니다.

![이호진 교수님과 함께하는 수학적 함수와 컴퓨터 프로그래밍 함수의 비교](./img/fig_02_02_b_math_vs_computer_function.jpg)  
*이호진 교수님이 정적인 수학적 대응 관계($f(x)=y$)와 동적인 컴퓨터 프로그램 함수의 연산 프로세스 차이를 설명하고 있습니다.*

---

구체적인 예가 있으면 좋겠군요. 

제곱을 계산하는 함수 $f(x) = x^2$ 이 있다고 해봅시다. 

이때 $y = f(x)$ 라고 하면 변수 $y$ 와 $x$ 의 관계가 함수 $f$ 에 의해 결정됩니다. 

즉, 함수 $f$ 에 의해 '$y$ 는 $x$ 의 제곱이다'라는 관계가 성립됩니다.

![제곱 연산 함수 f(x) = x^2의 수학적 입출력 대응 관계](./img/fig_02_03_square_math_relation.jpg)  
*도로시와 토토가 제곱 연산 함수 $y = f(x) = x^2$ 의 수학적 입출력 대응 관계를 살펴보고 있습니다.*

---

이와 같이 변수 사이의 대응 관계를 정하는 역할을 함수가 맡게 되며, [그림 2-1]은 그 의미를 시각적으로 표현한 모습입니다.

그림 2-1 변수와 함수의 관계
![그림 2-1](./img/fig-1321.jpg)

[그림 2-1]은 변수 $x$ 와 $y$, 그리고 함수 $f$ 의 관계를 보여줍니다. 

---

이처럼 원(○)과 사각형(□) 모양의 노드들을 화살표로 연결해 계산 과정을 표현한 그림을 계산 그래프<sup>computational graph</sup>라고 합니다. 

이 교제에서는 변수를 ○로, 함수를 □로 표시하겠습니다.

![계산 그래프 표기 규칙: 변수는 원(○), 함수는 사각형(□)](./img/fig_02_04_circle_variable_square_function.jpg)  
*계산 그래프 표기 규칙: 변수(Variable)는 원(○), 함수(Function)는 사각형(□)으로 표기합니다.*

---

> **CAUTION**  
> '그래프'라고 하면 막대그래프나 원그래프 같은 게 떠오를 수 있지만, 컴퓨터 과학에서 그래프<sup>graph</sup>는 노드<sup>node, 교점</sup>와 에지<sup>edge, 변, 간선</sup>로 구성된 데이터 구조(혹은 그 구조를 나타낸 그림)를 말합니다.
>
>  [그림 2-1]에서 화살표는 '방향이 있는' 에지에 해당합니다.

![노드와 유향 에지로 구성된 계산 그래프 구조](./img/fig_02_05_graph_theory_nodes_edges.jpg)  
*노드(원, 사각형)와 유향 에지(화살표)로 계산의 흐름을 표현하는 그래프 이론의 기본 구조입니다.*

---

## 2.2 Function 클래스 구현

그러면 [그림 2-1]의 함수를 프로그래밍 관점에서 생각해보죠. 

구체적으로는, 앞서 구현한 Variable 인스턴스를 변수로 다룰 수 있는 함수를 Function 클래스로 구현합니다. 

![Variable 인스턴스를 다루는 Function 클래스 설계 청사진](./img/fig_02_06_function_class_blueprint.jpg)  
*Variable 인스턴스를 입력받아 처리하는 Function 클래스의 설계 청사진입니다.*

---

이때 여기서 주의할 점은 다음 두 가지입니다.

* Function 클래스는 Variable 인스턴스를 입력받아 Variable 인스턴스를 출력합니다.
* Variable 인스턴스의 실제 데이터는 인스턴스 변수인 `data`에 있습니다.

![Function 클래스 설계의 2대 황금 원칙](./img/fig_02_07_function_two_golden_rules.jpg)  
*Function 클래스 설계의 2대 황금 원칙 (Variable 입출력 일관성 및 실제 데이터 참조)*

---

이 두 가지에 유의하여 Function 클래스를 다음과 같이 코드를 구현합니다.

```python
class Function:
    def __call__(self, input):
        x = input.data  # 데이터를 꺼낸다.
        y = x ** 2  # 실제 계산
        output = Variable(y)  # Variable 형태로 되돌린다.
        return output
```

---

`__call__` 메서드의 인수 input은 Variable 인스턴스라고 가정합니다.

따라서 실제 데이터는 `input.data`에 존재합니다. 

데이터를 꺼낸 후 원하는 계산(여기서는 제곱)을 하고, 결과를 Variable이라는 '상자'에 담아 돌려줍니다.

![조교 지니의 3단계 파이프라인: 언패킹, 연산, 리패킹](./img/fig_02_07_b_unpacking_processing_packing.jpg)  
*조교 지니가 `__call__` 내부에서 데이터를 꺼내고(Unpacking), 계산한 뒤 다시 Variable 상자에 포장(Packing)하는 3단계 과정을 안내하고 있습니다.*

---

> **NOTE**  
> `__call__` 메서드는 파이썬의 특수 메서드입니다. 
>
> 이 메서드를 정의하면 `f = Function()` 형태로 함수의 인스턴스를 변수 f에 대입해두고, 나중에 `f(...)` 형태로 `__call__` 메서드를 호출할 수 있습니다.

![파이썬 __call__ 특수 메서드의 호출 마법](./img/fig_02_08_python_call_dunder_magic.jpg)  
*파이썬 특수 메서드 `__call__`을 통해 Function 인스턴스를 마치 일반 수학 함수 $f(x)$처럼 자연스럽게 호출하는 원리입니다.*

---

## 2.3 Function 클래스 이용

Function 클래스를 실제로 사용해보죠. 

Variable 인스턴스인 $x$ 를 Function 인스턴스인 $f$ 에 입력해보겠습니다.

```python
x = Variable(np.array(10))
f = Function()
y = f(x)

print(type(y))  # type() 함수는 객체의 클래스를 알려준다.
print(y.data)
```

---

이와 같이 Variable과 Function을 연계할 수 있습니다. 

**실행 결과**

```
<class '__main__.Variable'>
100
```

실행 결과를 보면 $y$ 의 클래스는 Variable이며, 데이터는 `y.data`에 잘 저장되어 있음을 알 수 있습니다.

![도로시와 토토의 첫 번째 함수 실행 성공 검증](./img/fig_02_08_b_first_function_execution.jpg)  
*입력 상자 $x(10)$가 함수 $f$를 통과하여 데이터 $100$을 품은 새로운 변수 상자 $y$로 완성된 첫 번째 실행 결과를 기뻐하고 있습니다.*

---

그런데 방금 구현한 Function 클래스는 용도가 '입력값의 제곱'으로 고정된 함수입니다. 

![하드코딩된 계산 로직의 확장성 한계](./img/fig_02_09_hardcoded_function_limitation.jpg)  
*함수 클래스 내부에 제곱 로직이 고정되어 있어 다양한 수학 함수로 확장하기 어려운 한계점을 짚어봅니다.*

---

따라서 Function 클래스 이름 보다는 Square라는 명확한 이름이 더 어울립니다. 

앞으로 Sin, Exp 등 다양한 함수가 필요하다는 점을 고려하면 Function 클래스는 기반 클래스로 두고 DeZero의 모든 함수가 공통적으로 제공하는 기능만 담아두는 것이 좋겠습니다. 

![모듈식 기반 클래스로서의 Function 아키텍처](./img/fig_02_10_base_class_modular_architecture.jpg)  
*모든 연산이 공통으로 공유할 모듈식 기반 클래스(Base Class)로서의 Function 아키텍처입니다.*

---

그래서 앞으로 모든 DeZero 함수를 설계할때는 다음의 두 사항을 충족하도록 구현하겠습니다.

* Function 클래스는 기반 클래스로서, 모든 함수에 공통되는 기능을 구현합니다.
* 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현합니다.

![기반 클래스와 서브클래스의 객체지향 상속 트리](./img/fig_02_11_inheritance_principles_tree.jpg)  
*객체 지향 상속 원리: 기반 클래스(Function)와 구체적 파생 클래스(Square, Exp)의 상속 트리 계층 구조입니다.*

---

이를 위해 Function 클래스를 다음처럼 수정합니다.

**steps/step02.py**
```python
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)  # 구체적인 계산은 forward 메서드에서 한다.
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()
```

---

`__call__`을 살짝 수정하고 `forward`라는 메서드를 추가했습니다.

`__call__` 메서드는 'Variable에서 데이터 찾기'와 '계산 결과를 Variable에 포장하기'라는 두 가지 일을 합니다. 

그리고 그 사이의 구체적인 계산은 `forward` 메서드를 호출하여 수행합니다. 

![템플릿 메서드 패턴: __call__에서 forward로의 위임](./img/fig_02_12_forward_method_delegation.jpg)  
*템플릿 메서드 패턴: `__call__`에서 입출력 포장을 제어하고 구체적인 수식 연산은 `forward` 메서드로 위임합니다.*

---

마지막으로 `forward` 메서드의 구체적인 로직은 하위 클래스에서 구현합니다. 마치 템플릿 메소드 패턴 형태로 구현하는 것과 같습니다.

> **NOTE**  
> Function 클래스의 `forward` 메서드는 예외를 발생시킵니다. 
>
> 이렇게 해두면 Function 클래스의 `forward` 메서드를 직접 호출한 사람에게 '이 메서드는 상속하여 구현해야 한다'는 사실을 알려줄 수 있습니다.

![하위 클래스 구현을 강제하는 NotImplementedError 안전장치](./img/fig_02_13_not_implemented_safety_gate.jpg)  
*하위 클래스에서의 `forward` 구현을 강제하고 런타임 안전성을 보장하는 `NotImplementedError` 안전장치입니다.*

---

이어서 Function 클래스를 상속하여 입력값을 제곱하는 클래스를 구현하겠습니다. 

클래스 이름은 Square라고 짓고 다음과 같이 다시 구현해 보도록 합니다.

![Square 서브클래스 구현 완료](./img/fig_02_14_square_subclass_eureka.jpg)  
*Function을 상속받아 `forward` 메서드에 $x^2$만 구현하여 깔끔하게 완성된 Square 서브클래스입니다.*

---

**steps/step02.py**

```python
class Square(Function):
    def forward(self, x):
        return x ** 2
```

Square 클래스는 Function 클래스를 상속하기 때문에 `__call__` 메서드는 그대로 계승됩니다. 

따라서 `forward` 메서드에 구체적인 계산 로직을 작성해 넣는 것만으로 구현은 끝입니다. 

![로봇 틴맨과 도로시의 모듈식 상속 퍼즐 블록 완성](./img/fig_02_14_b_square_forward_execution.jpg)  
*로봇 틴맨과 도로시가 기반 클래스(Function) 틀 위에 Square 블록을 장착하여 단 2줄로 동작하는 모듈식 상속을 완성하고 있습니다.*

---

실제로 잘 동작하는지 Square 클래스를 사용하여 Variable을 처리하는 모습을 보시죠.

**steps/step02.py**
```python
x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
```

---

**실행 결과**

```
<class '__main__.Variable'>
100
```

보다시피 이전과 같은 결과를 얻었습니다. 

이것으로 2단계도 끝! 벌써 Variable과 Function 클래스의 기초가 완성되었습니다!

---

> **NOTE**  
> 한동안 Function의 입력과 출력은 '하나의 변수'로 한정합니다.
>
>  DeZero가 여러 변수를 다룰 수 있도록 확장하는 작업은 11단계에 가서 하겠습니다.

![단일 변수·함수 완성 및 11단계 다변수 확장 로드맵](./img/fig_02_15_single_variable_roadmap_step11.jpg)  
*단일 입출력 함수의 성공적인 완성과 향후 11단계 다변수 확장으로 이어질 DeZero 로드맵입니다.*
