# Lista_04_aprendizagem_de_maquina_UFC_mestrado

Uma rede **MLP (Multilayer Perceptron)** é uma arquitetura de rede neural artificial do tipo *feedforward* (onde a informação caminha apenas para a frente). Ela é composta por neurônios artificiais organizados em camadas estruturadas, projetada para aprender aproximações de funções não-lineares complexas.

A sua anatomia e o fluxo matemático do seu algoritmo, é dividido em três pilares: **A Estrutura**, o **Forward Propagation (Predição)** e o **Backpropagation (Aprendizado)**.

## 1. A Anatomia da MLP (As Camadas)

Uma MLP padrão possui três tipos de camadas organizadas sequencialmente:

 * **Camada de Entrada (Input Layer):** Não realiza cálculos. Ela simplesmente recebe as variáveis preditoras do mundo real (os 8 atributos do concreto ou os 18 do veículo) e os repassa para a camada seguinte.

 * **Camada Oculta (Hidden Layer):** É o "cérebro" da rede. É aqui que os neurônios combinam os dados de entrada usando pesos e aplicam uma função de ativação não-linear. Essa não-linearidade permite que a rede aprenda curvas e superfícies de decisão complexas (Teorema da Aproximação Universal).

 * **Camada de Saída (Output Layer):** Traduz o processamento da camada oculta na resposta final do problema (um valor contínuo na Regressão ou probabilidades na Classificação).

### O Neurônio Individual e o Bias
Cada neurônio individual dentro dessas camadas realiza uma operação simples dividida em duas etapas:

 1. **Combinação Linear:** Ele multiplica cada entrada pelo seu respectivo peso (w) e soma tudo, adicionando uma constante chamada **Bias** (b). O bias funciona como o coeficiente linear de uma reta, permitindo deslocar a função de ativação para a esquerda ou direita para ajustar melhor os dados.
   
 2. **Função de Ativação (\phi):** O resultado u (potencial de ativação) passa por uma função não-linear (como a Sigmoide ou ReLU). Sem essa função, a rede inteira seria apenas uma enorme equação linear (uma linha reta), incapaz de resolver problemas complexos.

## 2. Forward Propagation (Passo de Ida)

O *Forward Propagation* é o processo onde os dados entram pela camada de entrada, são processados e geram uma previsão na saída. Pensando no código matricial NumPy que usamos, o fluxo para uma amostra é:

```
  [Entrada X] 
       │
       ▼
 1. Multiplica pelos pesos W e soma o Bias  ──►  u = W·X + b
       │
       ▼
 2. Aplica a Ativação Oculta               ──►  z = Sigmoide(u)
       │
       ▼
 3. Multiplica pelos pesos M e soma o Bias  ──►  r = M·z + b
       │
       ▼
 4. Aplica a Ativação de Saída             ──►  o = Ativação_Saída(r)
       │
       ▼
  [Previsão Final O]

```

## 3. Backpropagation e SGD com Momentum (Passo de Volta)

Se o *Forward* é a rede tentando adivinhar a resposta, o *Backpropagation* é o algoritmo calculando o tamanho do erro e descobrindo como corrigir os pesos para não errar da próxima vez.

### Passo A: Medindo o Erro (Função de Custo)

A rede compara a previsão gerada (o) com a resposta real esperada (y) através de uma função de custo (MSE para regressão ou Entropia Cruzada para classificação).

### Passo B: Descobrindo a Culpa (Regra da Cadeia e Gradientes \delta)

Usando derivadas matemáticas (Regra da Cadeia), o algoritmo calcula o **Gradiente**. O gradiente aponta a direção que faz o erro *aumentar*. Como nós queremos *diminuir* o erro, nós vamos andar na direção oposta ao gradiente (Gradiente Descendente).

 * **Gradiente Local da Saída (\delta_{saida}):** Calcula o erro direto na ponta final (o - y).

 * **Gradiente Local Oculto (\delta_{oculta}):** Distribui o erro da saída de volta para os neurônios ocultos, ponderando pelo peso que cada um teve na linha de frente.

### Passo C: Atualização com SGD e Momentum

O Gradiente Descendente Estocástico (SGD) tradicional atualiza os pesos puramente subtraindo o gradiente multiplicado pela Taxa de Aprendizado (\eta).

O **Momentum**, funciona como uma simulação física de gravidade/inércia. Em vez de mudar o peso abruptamente a cada lote (*minibatch*), o algoritmo calcula uma **Velocidade (V)**:

 1. A nova velocidade recebe uma fração da velocidade antiga (coeficiente \beta) + o novo empurrão do gradiente atual:
   
 2. O peso é atualizado subtraindo essa velocidade acumulada:
   
> Se o gradiente estiver apontando consistentemente para a mesma direção ao longo das épocas, a velocidade aumenta (como uma bola rolando uma montanha abaixo), fazendo a rede convergir muito mais rápido. Se o gradiente encontrar um buraco raso (um mínimo local falso), a "inércia" da velocidade faz a rede passar direto por ele até encontrar o ponto de erro mínimo real.