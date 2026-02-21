# 📉 Churn Prediction com TensorFlow.js (Node)

Modelo simples de **classificação binária** para prever se um cliente vai **cancelar (`vai_cancelar`)** ou **ficar (`vai_ficar`)**, utilizando rede neural densa com TensorFlow.js no ambiente Node.js.

---------------------------------------------------------------------------------------------------

## 🚀 Objetivo

Este projeto demonstra:

- ✅ Criação de modelo com `tf.sequential`
- ✅ Normalização de dados (Min-Max)
- ✅ One-Hot Encoding
- ✅ Geração de dataset sintético
- ✅ Treinamento supervisionado
- ✅ Predição com probabilidade

> 🎓 Projeto educacional para entender o fluxo completo de um modelo de Machine Learning em Node.

---------------------------------------------------------------------------------------------------

## 🧠 Arquitetura do Modelo

### 🔢 Entrada

Total de **7 features**:

```
[
  tempo_cliente,
  logins_semana,
  tickets_suporte,
  plano_basic,
  plano_medium,
  plano_premium,
  atraso_pagamento
]
```

---------------------------------------------------------------------------------------------------

### 🏗 Estrutura da Rede Neural

```
Input (7)
   ↓
Dense (16 neurônios, ReLU)
   ↓
Dense (8 neurônios, ReLU)
   ↓
Dense (1 neurônio, Sigmoid)
```

---------------------------------------------------------------------------------------------------

## ⚙️ Configuração

| Parâmetro   | Valor |
|------------|--------|
| Otimizador | Adam (0.001) |
| Loss       | binaryCrossentropy |
| Métrica    | accuracy |
| Epochs     | 200 |

---------------------------------------------------------------------------------------------------

## 📊 Lógica de Negócio (Churn Artificial)

A probabilidade de churn aumenta quando:

- 📉 Cliente tem pouco tempo de contrato  
- 📉 Faz poucos logins  
- 📉 Abre muitos tickets  
- 📉 Está com pagamento em atraso  

A label é gerada usando uma **função logística probabilística**.

---------------------------------------------------------------------------------------------------

## 📦 Instalação

### 1️⃣ Setup
Utilize Node v22 para que não haja conflito com o Tensorflow.

### 2️⃣ Clonar o repositório

```bash
git clone <repo-url>
cd <repo>
```

### 3️⃣ Instalar dependências

```bash
npm install
```

Instale o TensorFlow para Node:

```bash
npm install @tensorflow/tfjs-node@4.22
```

---------------------------------------------------------------------------------------------------

## ▶️ Executar o projeto

Se estiver usando **TypeScript**:

```bash
npx ts-node index.ts
```

Ou compile:

```bash
tsc
node dist/index.js
```

---------------------------------------------------------------------------------------------------

## 🔎 Exemplo de Previsão

Cliente utilizado no exemplo:

```ts
{
  tempo_cliente: 25,
  logins_semana: 10,
  tickets_suporte: 18,
  plano: 'premium',
  atraso_pagamento: true
}
```

### 📤 Saída esperada:

```
Resultado da previsão:
Probabilidade de cancelar: 87.42%
Classificação: vai_cancelar
```

---------------------------------------------------------------------------------------------------

## 🧮 Normalização

Utiliza **Min-Max Scaling**:

```
(valor - min) / (max - min)
```

Limites definidos por regra de negócio:

| Feature          | Min | Max |
|------------------|-----|-----|
| tempo_cliente    | 0   | 60  |
| logins_semana    | 0   | 50  |
| tickets_suporte  | 0   | 20  |

---------------------------------------------------------------------------------------------------

## 🏷 Tipagem

```ts
type Plano = 'basic' | 'medium' | 'premium'

interface ClienteInput {
  tempo_cliente: number
  logins_semana: number
  tickets_suporte: number
  plano: Plano
  atraso_pagamento: boolean
}
```

---------------------------------------------------------------------------------------------------

## 📚 Conceitos Demonstrados

- Redes neurais densas  
- Classificação binária  
- Sigmoid  
- Cross Entropy  
- Normalização  
- One-hot encoding  
- Geração de dataset sintético  
- Probabilidade logística  

---------------------------------------------------------------------------------------------------

## 🛠 Possíveis Melhorias

- Separar treino e validação  
- Salvar e carregar modelo  
- Usar dataset real  
- Implementar early stopping  
- Criar API REST para servir o modelo  
- Adicionar testes unitários  

---------------------------------------------------------------------------------------------------

## 📌 Observação

Este projeto gera **dados sintéticos apenas para fins educacionais**.  
Não deve ser utilizado em produção sem validação adequada.

---------------------------------------------------------------------------------------------------

## 👨‍💻 Autor

Projeto desenvolvido para fins de estudo em Machine Learning aplicado com Node.js.